from typing import Union, List
import torch
from torch import nn
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmnotebook
import numpy as np

class HAPredictor:
    def __init__(
        self,
        dataset,
        N,
        predict_window,
    ):
        '''
        Parameters
        ----------
        dataset : GraphDataset
        N : int
            number of weeks to consider
        predict_window : int
            number of weeks to predict
        '''
        self.dataset = dataset
        self.N = N  # number of weeks to consider
        if predict_window is None or predict_window == -1:
            self.predict_window = len(dataset) // (240*7) - N
        else:
            self.predict_window = predict_window

    def get_past_datapoints(self, index):
        past_indices = [index - 240 * 7 * week for week in range(1, self.N + 1)]
        past_datapoints = [self.dataset[i].x for i in past_indices]
        stacked_past = torch.stack(past_datapoints).numpy()
        
        return stacked_past

    def predict(self):
        predictions = []
        crit = ['MAE', 'RMSE']
        errors = {k: torch.empty(0, device='cpu', dtype=torch.float16)
                 for k in crit}
        
        ds_start = 240*7*self.N
        ds_end   = 240*7*(self.N+self.predict_window)
        
        tqdm_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},'
                '{rate_fmt}{postfix}]')
        tqdm_args = {'total': ds_end, 'leave': True, 'desc': 'Epochs',
                      'colour': 'blue', 'bar_format': tqdm_fmt, 'initial': ds_start}
        
        # Initialize the progress bars
        pbar = tqdmnotebook(**tqdm_args)
        epoch_handle = display(pbar, display_id='pbar')

        # Start from the Nth week to ensure we have past data
        # for i in range(240 * 7 * self.N, len(self.dataset)):
        for i in range(ds_start, ds_end):
            past_datapoints = self.get_past_datapoints(i)
            prediction = np.mean(past_datapoints, axis=0)
            actual = self.dataset[i].x.numpy()
            for c in crit:
                errors[c] = torch.cat(
                    [
                        errors[c],
                        self.calculate_crit(c, actual, prediction)
                    ],
                dim=0).detach()
            pbar.update(1)
            
        return errors
                                       
    @staticmethod
    def calculate_crit(crit, y, y_hat):
        '''
        Calculate various error metrics between actual and predicted values.
        This method supports MAE, MSE, MAPE, and RMSE.
    
        Parameters
        ----------
        crit: {'MAE', 'MSE', 'MAPE', 'RMSE'}
            the criterion to use for calculating the error
        y: np.ndarray
            the actual values, of shape (N, *) with N being the batch size
        y_hat: np.ndarray
            the predicted values, of same shape as `y`
    
        Returns
        -------
        torch.Tensor
            the calculated error based on the specified criterion, of shape
            (N, )
        '''
        y = torch.tensor(y)
        y_hat = torch.tensor(y_hat)
        if crit == 'MAE':
            return torch.mean(torch.abs(y - y_hat), dim=1)
        elif crit == 'MSE':
            return torch.mean(torch.square(y - y_hat), dim=1)
        elif crit == 'MAPE':
            epsilon = 1e-8  # Small constant to avoid division by zero
            return torch.mean(torch.abs((y - y_hat) / (y + epsilon)), dim=1) * 100
        elif crit == 'RMSE':
            return torch.sqrt(torch.mean(torch.square(y - y_hat), dim=1))

        
class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        bias: bool,
    ):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        ) * 2


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
        B: batch
        T: seq_len
        C: channels
        H: height
        W: width
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super().__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        assert (
            len(kernel_size) == len(hidden_dim) == num_layers
        ), "Inconsistent list length."

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list[-1], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers: int):
        if isinstance(param, int):
            return (param,) * num_layers
        elif isinstance(param, list):
            return tuple(param)
        else:
            return param
        
class ConvLSTMPredictor(nn.Module):
    def __init__(
        self, 
        num_channels: int, 
        hidden_dims: List[int],
        kernel_size: Union[int, List[int]],
        num_layers:int=1,
        batch_first:bool=True,
        bias:bool=True,
        return_all_layers:bool=False,
        prediction_window=[0, 1, 2, 5, 8, 11],
        dropout=0.5
    ):
        super(ConvLSTMPredictor, self).__init__()
        input_dims = [num_channels] + [el for el in hidden_dims[1:-1:2]]
        hidden_dims = [hidden_dims[k:k+2] for k in range(0, len(hidden_dims), 2)]
        self.num_layers=len(input_dims)
        self.seq = nn.Sequential()
        for i in range(len(input_dims)):
            conv = ConvLSTM(
                input_dim=input_dims[i],
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_size,
                num_layers=len(hidden_dims[i]),
                batch_first=True,
                bias=True,
                return_all_layers=False,)
            relu = nn.ReLU()
            self.seq.append(conv)
            self.seq.append(relu)

        self.linear = nn.Linear(
            in_features=hidden_dims[-1][-1],
            out_features=num_channels
        )
        
        self.prediction_window = prediction_window
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        for i in range(self.num_layers):
            x, _ = self.seq[2*i](x)
            x = self.seq[2*i + 1](x)
            
        x = torch.movedim(x, 2, -1)
        x = self.linear(x)        
        x = torch.movedim(x, -1, 2)
        
        x = x[:, self.prediction_window, ...]
        x = self.dropout(x)
        
        x = self.sigmoid(x)
        
        return x