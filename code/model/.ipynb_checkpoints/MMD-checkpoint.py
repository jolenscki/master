'''
As we don't want/need to backtrack any of the variables of these functions or
classes, we add @torch.no_grad() at them.
'''

import torch
from torch import nn
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

class RBF(nn.Module):
    """
    Radial Basis Function (RBF) kernel module.

    Parameters
    ----------
    num_kernels : int
        Number of RBF kernels to use.
    bandwidth : float, optional
        Bandwidth parameter for the RBF kernel. If not provided, it is computed
        as the median pairwise squared Euclidean distance between the inputs.
    eps : float, optional
        Small value to add to the diagonal of the kernel matrix for numerical stability.
    """
    def __init__(self, num_kernels=5, bandwidth=None, eps=1e-2):
        super().__init__()
        self.num_kernels = num_kernels
        self.bandwidth = bandwidth
        self.eps = eps
        
    @torch.no_grad()
    def forward(self, X, Y):
        """
        Compute the RBF kernel matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        Y : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        K : torch.Tensor
            RBF kernel matrix of shape (batch_size, batch_size).
        """
        X = torch.tanh(X)
        Y = torch.tanh(Y)
        # Compute pairwise squared Euclidean distances
        XX = torch.sum(X**2, dim=1, keepdim=True)
        YY = torch.sum(Y**2, dim=1, keepdim=True)
        XY = torch.einsum('bij,bik->bjk', X, Y)
        
        # Define bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = torch.median(XX) + torch.median(YY) - 2 * torch.median(XY)

        # Calculate the RBF kernel
        pre_exp = XX - 2 * XY + YY
        threshold = 88
        div = torch.div(pre_exp, self.bandwidth + self.eps)
        div = torch.clamp(div, max=threshold)   
        K = torch.exp(-div)

        return K
    

class MMDLoss(nn.Module):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between two sets of samples X and Y.
    The MMD loss is a measure of the distance between the distributions of X and Y.

    Parameters
    ----------
    kernel : callable, optional
        The kernel function to use for computing the MMD loss. Default is RBF().
    num_kernels : int, optional
        The number of kernels to use for computing the MMD loss. Default is 5.
    batch_size : int, optional
        The batch size to use for computing the MMD loss. Default is 32.
    logging : bool, optional
        Whether to log/print the MMD loss. Default is False.
    """
    def __init__(self,
                kernel=RBF(), 
                num_kernels=5, 
                batch_size=32,
                logging: bool = False):
        super().__init__()
        self.kernel = kernel
        self.num_kernels = num_kernels
        self.batch_size = batch_size
        self.logging = logging
        
    @torch.no_grad()
    def forward(self, X, Y):
        """
        Compute the MMD loss.

        Parameters
        ----------
        X : tensor
            The first set of samples.
        Y : tensor
            The second set of samples.

        Returns
        -------
        float
            The MMD loss.
        """
        X = torch.movedim(X, 1, 0)
        Y = torch.movedim(Y, 1, 0)
        batch_size = self.batch_size
        min_samples = min(X.shape[0], Y.shape[0])
        indices_x = torch.randperm(X.shape[0])[:min_samples]
        indices_y = torch.randperm(Y.shape[0])[:min_samples]
        X_ = X[indices_x, ...]
        Y_ = Y[indices_y, ...]
        num_samples_x = X_.size(0)
        num_samples_y = Y_.size(0)
        if batch_size == -1:
            x_mini_batches = [X_]
            y_mini_batches = [Y_]
        else:
            # Generate random mini-batches for X and Y
            x_mini_batches = [X_[i:i+batch_size] for i in range(0, num_samples_x - batch_size, batch_size)]
            y_mini_batches = [Y_[i:i+batch_size] for i in range(0, num_samples_y - batch_size, batch_size)]

        # Initialize loss
        mmd_loss = 0

        start_time = time.time()
        # Calculate MMD loss for each mini-batch
        for i, x_mini_batch in enumerate(x_mini_batches):
            for j, y_mini_batch in enumerate(y_mini_batches):
                K = self.kernel(x_mini_batch, y_mini_batch)
                mmd_loss += K.mean()
        if self.logging:
            print(f'Elapsed time: {time.time() - start_time:.2f} seconds')

        # Average the loss over mini-batches
        mmd_loss /= (len(x_mini_batches) * len(y_mini_batches))

        # Check if loss is infinite or NaN
        if (torch.isinf(mmd_loss).any() or torch.isnan(mmd_loss).any()):
            print(f"{torch.isinf(x_mini_batch).any()}")
            print(f"{torch.isinf(y_mini_batch).any()}")
            print("MMD loss is inf or NaN")

        return mmd_loss