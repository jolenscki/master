import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

import random
import os
import os.path as osp
import sys
from tqdm.notebook import tqdm as tqdmnotebook

from model.autoencoder import Autoencoder
from model.DAN import GradientReversalLayer, DomainDiscriminator
from model.predictor import Predictor

from util.plot_utils import *
from model.criterions import *

class Model(nn.Module):
    '''
    Full model class, comprises the instantiation of all components
    (autoencoder, domain discriminator, and predictor), with training scripts
    and testing methods

    Methods
    -------
    ae_train
        organize and execute the autoencoder training
    ae_test
        test the trained autoencoder
    dd_test
        test the domain discriminator 
    pred_train
        organize and execute the predictor training
    pred_test
        test the trained predictor
    save_module
        save an specific module ('autoencoder', 'discriminator', or 
        'predictor') as a *.pth file
    load_module
        load an specific module ('autoencoder', 'discriminator', or
        'predictor') from a *.pth file
    calculate_crit
        calculate a metric ('MAE', 'MAPE', 'MSE', 'RMSE')
    fusion_features
        to be used within the predictor training/testing; executes the fusion
        between the data features (ST) and time (dow, hod) features
    '''
    def __init__(
        self,
        AE_parameters,
        DD_parameters,
        PR_parameters,
        num_epochs,
        dataloaders,
        AE_criterion,
        PR_criterion,
        optimizer_parameters,
        BATCH_SIZE,
        dd_lambda,
        folder,
        specs,
        tgt='MELBOURNE',
        val_dl=None,
        lr_scheduler=None
                ):
        '''
        Parameters
        ----------
        AE_parameters: dict
            parameters to instantiate the autoencoder module
        DD_parameters: dict
            parameters to instantiate the domain discriminator
        PR_parameters: dict
            parameters to instantiate the predictor
        num_epochs: Union[int, list]
            number of training epochs; if is an int, we use the same value for
            both the ae/dd and predictor training, if it's a list, we use
            different values
        dataloaders: set[torch_geometric.Dataloader]
            set containing train (sources), train (target), and test (target)
            dataloaders
        AE_criterion: nn.Module
            criterion to be used as loss function during autoencoder training
        PR_criterion: nn.Module
            criterion to be used as loss function during predictor training
        optimizer_parameters: Union[set, list[set]]
            parameters for the optimizer(s); if a list is passed, we interpret
            it as being the list of parameters for the autoencoder and 
            predictor optimizers; if a set is passed, we assume that the 
            parameters are to be the same for both training scripts
        BATCH_SIZE: int
            size of the batches in the dataloaders
        dd_lambda: float
            regularizator between domain discriminator and reconstruction
            losses on the autoencoder training
        folder: str
            folder in which the models are to be saved/loaded from
        specs: str
            unique identifier of the experiment on the model
        tgt: str, default='MELBOURNE'
            name of the target city
        val_dl: torch_geometric.Dataloader, default=None
            validation dataset dataloader
        lr_scheduler: set(int, float), default=None
            parameters to use in the learning rate scheduler (StepLR)     
        '''
        super(Model, self).__init__()
        self.ae = Autoencoder(**AE_parameters)
        self.GRL = GradientReversalLayer()
        self.dd = DomainDiscriminator(**DD_parameters)
        self.pred = Predictor(**PR_parameters)
        self.pt_dl, self.ft_dl, self.test_dl = dataloaders
        if val_dl is not None:
            self.val_dl = val_dl
        
        self.AE_criterion = AE_criterion
        self.PR_criterion = PR_criterion
        self.EPOCHS = num_epochs
        if not isinstance(optimizer_parameters[0], list):
            optimizer_parameters = [
                optimizer_parameters, optimizer_parameters
            ]
        self.ae_lr, self.ae_l2_decay = optimizer_parameters[0]
        self.pred_lr, self.pred_l2_decay = optimizer_parameters[1]
        self.BATCH_SIZE = BATCH_SIZE
        self.dd_lambda = dd_lambda
        self.folder = folder
        self.specs = specs
        self.tgt = tgt
        self.lr_scheduler = lr_scheduler
    
    def ae_train(
        self, 
        mode, 
        lambda_update=1, 
        save=True, 
        plot=None,
        accumulation_steps=32
    ):
        '''
        Train the autoencoder model along with the domain discriminator. This
        method performs training of the autoencoder (`ae`) and domain 
        discriminator (`dd`) using either pretraining or finetuning modes. It 
        allows for gradient accumulation, updating the learning rate, and 
        plotting of losses.
    
        Parameters
        ----------
        mode: {'pretrain', 'finetune'}
            specifies the training mode
        lambda_update: float, default=1
            the factor by which the domain discriminator loss weight (lambda) 
            is updated after each epoch
        save: bool, default=True
            indicates whether to save the trained models
        plot: Union[Any, None], default=None
            whether to plot the training losses after the training script 
        accumulation_steps: int, default32
            the number of steps to accumulate gradients before performing an 
            optimizer step
        '''
        if mode == 'pretrain':
            lr=self.ae_lr
            l2_decay=self.ae_l2_decay
            dl=self.pt_dl
        elif mode == 'finetune':
            lr=self.ae_lr/2
            l2_decay=self.ae_l2_decay/2
            dl=self.ft_dl
        
        optimizer = torch.optim.AdamW(
            [
                {'params': self.ae.parameters()},
                {'params': self.dd.parameters()},
            ], lr=lr, weight_decay=l2_decay
        )
        if self.lr_scheduler is not None:
            step_size, gamma = self.lr_scheduler
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        scaler = GradScaler()
        tqdm_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},'
                '{rate_fmt}{postfix}]')
        tqdm_epoch = {'total': self.EPOCHS, 'leave': True, 'desc': 'Epochs',
                      'colour': 'blue', 'bar_format': tqdm_fmt}
        tqdm_batch = {'total': dl._dataset_length()//self.BATCH_SIZE + 1,
                      'leave': True, 'desc': 'Batches', 'colour': 'green',
                      'bar_format': tqdm_fmt}
        
        # Initialize the progress bars
        pbar_epochs = tqdmnotebook(**tqdm_epoch)
        epoch_handle = display(pbar_epochs, display_id='pbar_epoch')
        pbar_batches = tqdmnotebook(**tqdm_batch)
        batch_handle = display(pbar_batches, display_id='pbar_batch')
        
        dd_lambda = self.dd_lambda
        train_losses = []
        rec_losses = []
        dd_losses = []
        
        self.ae.train()
        self.dd.train()
        encoder = self.ae.encoder
        decoder = self.ae.decoder
        
        for epoch in range(self.EPOCHS):
            optimizer.zero_grad()
            for databatch, i, _ in dl:
                total_loss = torch.tensor(0., device=self.ae.device)
                
                #####################################################
                src_cities = list(databatch.keys())
                src_cities.remove(self.tgt)
                if src_cities:
                    selected_city = random.choice(src_cities)
                else:
                    selected_city = self.tgt
                #####################################################
                
                for city, data in databatch.items():
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    with torch.cuda.amp.autocast():
                        H = encoder(x, edge_index)
                        
                        #############################################
                        if city in [selected_city, self.tgt] and dd_lambda > 0:
                                H = self.GRL(H)
                                dd_loss = self.dd(
                                    H, edge_index, batch, city, self.tgt
                                )
                                total_loss += dd_lambda * dd_loss
                        #############################################

                        x_recons = decoder(H, edge_index)
                        x = x.reshape(x_recons.shape)
                        rec_loss = self.AE_criterion(x_recons, x)
                        total_loss += rec_loss
                
                loss = total_loss / accumulation_steps
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward(retain_graph=False)
                if (i + 1) % accumulation_steps == 0 or (i+1) == dl._dataset_length()//self.BATCH_SIZE:
                    scaler.step(optimizer)
                    if self.lr_scheduler is not None:
                        scheduler.step()
                    scaler.update()               
                    train_losses.append(scaled_loss.item())
                    rec_losses.append(rec_loss.item())
                    if city in [src_cities, self.tgt] and dd_lambda > 0:
                        dd_losses.append(dd_loss.item())    
                    optimizer.zero_grad()
                
                pbar_batches.update(1)
                if i % (dl._dataset_length()//(self.BATCH_SIZE*30)) == 0 and i > 0:
                    print("Batch: ", i)
                    print(f"    Loss: {total_loss.item():9.4f}")
                    print(f"     Rec: {rec_loss.item():9.4f}")
                    loss_per_epoch(
                        train_losses, epoch+1, self.specs+'total', 
                        save=True, show=True, ma=False
                    )
                    loss_per_epoch(
                        rec_losses, epoch+1, self.specs+'rec', 
                        save=True, show=True, ma=False
                    )
                    if dd_lambda > 0:
                        print(f"    Disc: {dd_loss.item():9.4f}")
                        loss_per_epoch(
                            dd_losses, epoch+1, self.specs+'dd', 
                            save=True, show=True, ma=False
                        )
            pbar_batches.reset()
            pbar_epochs.update(1)
            dd_lambda *= lambda_update
        
        if plot is not None:
            loss_per_epoch(
                train_losses, self.EPOCHS, self.specs,
                save=True
            )
            loss_per_epoch(
                rec_losses, self.EPOCHS, self.specs,
                save=True
            )
            if dd_lambda > 0:
                loss_per_epoch(
                    dd_losses, self.EPOCHS, self.specs,
                save=True
                )
            
        if save:
            save_specs = self.specs.replace('.', '')
            self.save_module(
                'autoencoder', self.folder, f'ae_{save_specs}.pth'
            )
            self.save_module(
                'discriminator', self.folder, f'dd_{save_specs}.pth'
            )
        
    def ae_test(self, crit=['MAE', 'MSE'], sample_limit=-1):
        '''
        Evaluate the autoencoder's performance on the test dataset using 
        specified metrics. This method tests the autoencoder by reconstructing
        the input data and calculating specified error metrics (e.g., MAE, 
        MSE) on the test dataset. It processes the test data in batches and 
        aggregates the error metrics up to a specified sample limit.
    
        Parameters
        ----------
        crit: list[str], default=['MAE', 'MSE']
            a list of criteria (error metrics) to compute; supported values
            include 'MAE', 'MSE', 'MAPE', 'RMSE'
        sample_limit: Union[int, float], default=-1
            the limit on the number of samples (batches) to consider for 
            testing; if -1 (default), all samples in the test loader are used;
            if a float is provided, it is assumed to be the fraction of the 
            total samples
    
        Returns
        -------
        dict
            a dictionary where keys are the error metric names (as given in
            `crit`), and values are tensors representing the aggregated errors
            across all processed batches
        '''
        dl = self.test_dl
        if sample_limit == -1:
            sample_limit = dl._dataset_length//self.BATCH_SIZE
        elif isinstance(sample_limit, float):
            sample_limit = int((dl._dataset_length//self.BATCH_SIZE)*sample_limit)
        
        tqdm_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},'
                '{rate_fmt}{postfix}]')
        tqdm_batch = {'total': sample_limit,
                      'leave': True, 'desc': 'Batches', 'colour': 'green',
                      'bar_format': tqdm_fmt}
        self.ae.eval()
        self.GRL.eval()
        self.dd.eval()
        encoder = self.ae.encoder
        decoder = self.ae.decoder
        errors = {k: torch.empty(0, device=self.ae.device, dtype=torch.float16)
                 for k in crit}
        
        with tqdmnotebook(**tqdm_batch) as pbar:
            for databatch, i, _ in dl:
                for city, data in databatch.items():
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    batch_size = int(data.ptr.shape[0] - 1)
                    with torch.cuda.amp.autocast():
                        H = encoder(x, edge_index)
                        x_recons = decoder(H, edge_index)
                        x = x.reshape(batch_size, -1)
                        x_recons = x_recons.reshape(x.shape)
                    for c in crit:
                        errors[c] = torch.cat([
                            errors[c],
                            self.calculate_crit(c, x, x_recons)
                        ], dim=0).detach()
                
                if i > sample_limit:
                    break
                    
        return errors
    
    def dd_test(self,):
        '''
        Evaluate the domain discriminator's performance on the validation
        dataset. This method tests the domain discriminator (`dd`) by 
        predicting the domain of each sample in the validation dataset and 
        calculating the accuracy of these predictions. The method uses the 
        autoencoder's encoder for feature extraction and the Gradient
        Reversal Layer (GRL) for domain adaptation.
    
        Returns
        -------
        float
            The accuracy of the domain discriminator on the validation 
            dataset, calculated as the proportion of correctly classified 
            samples.
        '''
        correct = 0
        total   = 0
        dd = self.dd
        ae = self.ae
        dd.eval()
        ae.eval()
        with torch.no_grad():
            for databatch, i, _ in self.val_dl:
                for city, data in databatch.items():
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    H = ae.encoder(x, edge_index)
                    H = self.GRL(H)
                    logits = self.dd(
                        H, edge_index, batch, city, self.tgt,
                        return_logits=True
                    )
                    domain_pred = (F.sigmoid(logits) > .5).float()
                    if city == self.tgt:
                        domain_pred_truth = torch.zeros_like(domain_pred)
                    else:
                        domain_pred_truth = torch.ones_like(domain_pred)
                    total += domain_pred_truth.size(0)
                    correct += (domain_pred == domain_pred_truth).sum().item()
        acc =  correct / total

        dd.train()
        ae.train()
            
        return acc
    
    def pred_train(
        self,
        mode,
        save=True,
        plot=None,
        accumulation_steps=32,
    ):
        '''
        Train the predictor model in either pretraining or finetuning mode.
        This method handles the training of the predictor component of the 
        model, using specified learning rates and decay parameters for 
        pretraining and finetuning modes. It supports gradient accumulation, 
        learning rate scheduling, and optional loss plotting.
    
        Parameters
        ----------
        mode: {'pretrain', 'finetune'}
            specifies the training mode
        save: bool, default=True
            indicates whether to save the trained models
        plot: Union[Any, None], default=None
            whether to plot the training losses after the training script 
        accumulation_steps: int, default32
            the number of steps to accumulate gradients before performing an 
            optimizer step
        '''
        if mode == 'pretrain':
            lr=self.pred_lr
            l2_decay=self.pred_l2_decay
            dl=self.pt_dl
        elif mode == 'finetune':
            lr=self.pred_lr/2
            l2_decay=self.pred_l2_decay/2
            dl=self.ft_dl
        
        optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.pred.parameters(),
                }
            ], lr=lr, weight_decay=l2_decay
        )
        if self.lr_scheduler is not None:
            step_size, gamma = self.lr_scheduler
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        scaler = GradScaler()
        tqdm_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},'
                '{rate_fmt}{postfix}]')
        tqdm_epoch = {'total': self.EPOCHS, 'leave': True, 'desc': 'Epochs',
                      'colour': 'blue', 'bar_format': tqdm_fmt}
        tqdm_batch = {'total': dl._dataset_length()//self.BATCH_SIZE + 1,
                      'leave': True, 'desc': 'Batches', 'colour': 'green',
                      'bar_format': tqdm_fmt}
        
        # Initialize the progress bars
        pbar_epochs = tqdmnotebook(**tqdm_epoch)
        epoch_handle = display(pbar_epochs, display_id='pbar_epoch')
        pbar_batches = tqdmnotebook(**tqdm_batch)
        batch_handle = display(pbar_batches, display_id='pbar_batch')
        
        train_losses = []
        
        self.pred.train()
        for epoch in range(self.EPOCHS):
            optimizer.zero_grad()
            for databatch, i, _ in dl:
                total_loss = torch.tensor(0., device=self.pred.device)
                for city, data in databatch.items():
                    if mode == 'pretrain' and city == self.tgt:
                        continue
                    x, edge_index = data.x, data.edge_index
                    y, batch = data.y, data.batch
                    date_features = data.date_features
                    y /= 255
                    batch_size = int(data.ptr.shape[0] - 1)
                    with torch.cuda.amp.autocast():
                        H = self.ae.encoder(x, edge_index)
                        H = self.fusion_features(H, date_features, batch_size)
                        y_pred = self.pred(H, edge_index, batch)
                        y = y.reshape(y_pred.shape)
                        loss = self.PR_criterion(y, y_pred)
                    total_loss += loss

                loss = total_loss / accumulation_steps
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward(retain_graph=False)
                if (i + 1) % accumulation_steps == 0 or (i+1) == dl._dataset_length()//self.BATCH_SIZE:
                    scaler.step(optimizer)
                    if self.lr_scheduler is not None:
                        scheduler.step()
                    scaler.update()
                    train_losses.append(scaled_loss.item())
                    optimizer.zero_grad()
                pbar_batches.update(1)
                if i % (dl._dataset_length()//(self.BATCH_SIZE*30)) == 0 and i > 0:
                    print("Batch: ", i)
                    print(f"    Loss: {total_loss.item():9.4f}")
                    loss_per_epoch(
                        train_losses, epoch+1, self.specs,
                        save=True, show=True, ma=False
                    )
            pbar_batches.reset()
            pbar_epochs.update(1)
        
        if plot is not None:
            loss_per_epoch(
                train_losses, self.EPOCHS, self.specs, save=True
            )
            
        if save:
            save_specs = self.specs.replace('.', '')
            self.save_module('predictor', self.folder, f'pred_{save_specs}.pth')
            
    def pred_test(self, crit=['MAE', 'MSE'], sample_limit=-1):
        '''
        Perform a prediction test on the test dataset and compute specified 
        error metrics. This method iterates over the test data loader, 
        performs predictions using the model's predictor and encoder, and 
        computes error metrics such as MAE and MSE for each batch. It
        aggregates these errors across batches up to a specified limit.

        Parameters
        ----------
        crit: list[str], default=['MAE', 'MSE']
            a list of criteria (error metrics) to compute; supported values
            include 'MAE', 'MSE', 'MAPE', 'RMSE'
        sample_limit: Union[int, float], default=-1
            the limit on the number of samples (batches) to consider for 
            testing; if -1 (default), all samples in the test loader are used;
            if a float is provided, it is assumed to be the fraction of the 
            total samples
    
        Returns
        -------
        dict
            a dictionary where keys are the error metric names (as given in
            `crit`), and values are tensors representing the aggregated errors
            across all processed batches
        '''
        dl = self.test_dl

        if sample_limit == -1:
            sample_limit = dl._dataset_length//self.BATCH_SIZE
        elif isinstance(sample_limit, float):
            sample_limit = int((dl._dataset_length//self.BATCH_SIZE)*sample_limit)
            
        tqdm_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},'
                '{rate_fmt}{postfix}]')
        tqdm_batch = {'total': sample_limit,
                      'leave': True, 'desc': 'Batches', 'colour': 'green',
                      'bar_format': tqdm_fmt}
        
        # Initialize the progress bars
        pbar = tqdmnotebook(**tqdm_batch)
        epoch_handle = display(pbar, display_id='pbar')
        
        
        self.pred.eval()
        encoder = self.ae.encoder
        errors = {k: torch.empty(0, device=self.pred.device, dtype=torch.float16)
                 for k in crit}
        
        with torch.no_grad():
            for databatch, i, _ in dl:
                data = databatch[self.tgt]
                x, edge_index = data.x, data.edge_index
                y, batch = data.y, data.batch
                date_features = data.date_features
                y /= 255
                batch_size = int(data.ptr.shape[0] - 1)
                with torch.cuda.amp.autocast():
                    H = encoder(x, edge_index)
                    H = self.fusion_features(H, date_features, batch_size)
                    y_pred = self.pred(H, edge_index, batch)
                y_pred = y_pred.reshape(batch_size, -1)
                y = y.reshape(batch_size, -1)
                pbar.update(1)

                for c in crit:
                    errors[c] = torch.cat(
                        [errors[c], self.calculate_crit(c, y, y_pred)],
                        dim=0
                    ).detach()
                if i > sample_limit:
                    break
   
        return errors
        
    def save_module(self, module_name, folder, filename):
        '''
        Save the state dictionary of a specified module of the class into a
        `*.pth` file.
    
        Parameters
        ----------
        module_name: {'autoencoder', 'discriminator', 'predictor'}
            the name of the module from which the state dictionary will be
            saved
        folder: str
            the path to the directory where the state dictionary file will be
            saved
        filename: str
            the name of the file where the state dictionary, with `.pth`
            extension, will be saved
        '''
        module = {'autoencoder': self.ae,
                  'discriminator': self.dd,
                  'predictor': self.pred}[module_name]
        torch.save(module.state_dict(), osp.join(folder, filename))
        
    def load_module(self, module_name, folder, filename):
        '''
        Load the state dictionary into a specified module of the class.
    
        Parameters
        ----------
        module_name: {'autoencoder', 'discriminator', 'predictor'}
            the name of the module to which the state dictionary will be
            loaded
        folder: str
            the path to the directory containing the state dictionary file.
        filename: str
            the name of the file containing the state dictionary, with `*.pth`
            extension
        '''
        module = {'autoencoder': self.ae,
                  'discriminator': self.dd,
                  'predictor': self.pred}[module_name]
        module.load_state_dict(torch.load(osp.join(folder, filename)))
                    
    @staticmethod
    def calculate_crit(crit, y, y_hat):
        '''
        Calculate various error metrics between actual and predicted values.
        This method supports MAE, MSE, MAPE, and RMSE.
    
        Parameters
        ----------
        crit: {'MAE', 'MSE', 'MAPE', 'RMSE'}
            the criterion to use for calculating the error
        y: torch.Tensor
            the actual values, of shape (N, *) with N being the batch size
        y_hat: torch.Tensor
            the predicted values, of same shape as `y`
    
        Returns
        -------
        torch.Tensor
            the calculated error based on the specified criterion, of shape
            (N, )
        '''
        if crit == 'MAE':
            return torch.mean(torch.abs(y - y_hat), dim=1)
        elif crit == 'MSE':
            return torch.mean(torch.square(y - y_hat), dim=1)
        elif crit == 'MAPE':
            epsilon = 1e-8  # Small constant to avoid division by zero
            return torch.mean(torch.abs((y - y_hat) / (y + epsilon)), dim=1) * 100
        elif crit == 'RMSE':
            return torch.sqrt(torch.mean(torch.square(y - y_hat), dim=1))
        
    @staticmethod 
    def fusion_features(H, T, BATCH_SIZE):
        '''
        Fusion method for ST features and time (periodic day-of-week and 
        hour-of-day) features.
        
        Parameters
        ----------
        H: torch.Tensor
            ST features, of shape (seq_len, BATCH_SIZE*num_nodes, hidden_dim)
        T: torch.Tensor
            periodic day-of-week and hour-of-day features, of shape 
            (4*BATCH_SIZE, )
        BATCH_SIZE: int
            batch size
            
        Returns
        -------
        torch.Tensor
            fused features, of shape (seq_len, conv_dim, hidden_dim + 4)
        '''
        T_extended = T.unsqueeze(0).unsqueeze(0) # (1, 1, 4*BATCH_SIZE)
        
        
        seq_len, batch_num_nodes, hidden_dim = H.shape
        num_nodes = batch_num_nodes//BATCH_SIZE
        
        # shape: (1, BATCH_SIZE, 4)
        T_extended = T_extended.reshape(1, BATCH_SIZE, -1) 
        
        # shape: (1, BATCH_SIZE*num_nodes, 4)
        T_extended = T_extended.repeat(1, num_nodes, 1)
        
        # shape: (seq_len, BATCH_SIZE*num_nodes, 4)
        T_extended = T_extended.expand(seq_len, -1, -1)
        
        return torch.cat([H, T_extended], dim=-1)
    