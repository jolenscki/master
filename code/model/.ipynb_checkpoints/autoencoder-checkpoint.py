import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import TopKPooling, global_mean_pool
from torch_geometric_temporal import GConvLSTM

class Encoder(nn.Module):
    """
    Encoder module for a Convolutional LSTM Autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimension of the input tensor.
    conv_dim : int
        Dimension of the linear layer output.
    hidden_dim : int
        Dimension of the hidden layer output.
    num_channels : int
        Number of channels in the input tensor.
    device : torch.device, optional
        Device to run the module on (default: 'cuda' if available, else 'cpu').
    activation : str, optional
        Activation function to use (default: 'tanh').
    dropout : float, optional
        Dropout probability (default: 0.5).
    """
    def __init__(self,
                 conv_dim,
                 num_channels,
                 K_cheb=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 activation='tanh',
                 dropout=0.5,
                 linear_dim: int=None,
                 use_layer_norm=False
                ):
        super(Encoder, self).__init__()
        self.conv_dim        = conv_dim
        self.num_channels    = num_channels
        self.K_cheb          = K_cheb
        self.device          = device
        self.use_layer_norm  = use_layer_norm
        
        self.rnn_cell = GConvLSTM(
            in_channels=num_channels, 
            out_channels=conv_dim,
            K=self.K_cheb,
        )
        if use_layer_norm:
            self.bn         = nn.LayerNorm(conv_dim)
        else:
            self.bn         = nn.BatchNorm1d(conv_dim)
        self.activation = self.switch_activation(activation)
        self.dropout    = nn.Dropout(dropout)
        self.linear     = None
        if linear_dim is not None:
            # self.linear = nn.Linear(conv_dim, linear_dim)
            hidden_dim = (conv_dim + linear_dim) // 2
            self.linear = nn.Sequential(
                nn.Linear(conv_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, linear_dim)
            )

    def forward(self, x, edge_index, h=None, c=None):
        """
        Forward pass of the Encoder module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (num_nodes, num_channels, seq_len).
        edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges).
        h : torch.Tensor, optional
            Hidden state tensor of shape (num_nodes, hidden_dim) (default: None).
        c : torch.Tensor, optional
            Cell state tensor of shape (num_nodes, hidden_dim) (default: None).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (seq_len, conv_dim, hidden_dim).
        """
        num_nodes = x.shape[0]
        x = x.reshape(num_nodes, self.num_channels, -1)
        
        H = []
        for t in range(x.shape[-1]):
            x_t = x[..., t].reshape(num_nodes, self.num_channels)
            h, c = self.rnn_cell(x_t, edge_index, None, H=h, C=c)
            h_t, c_t = h.detach(), c.detach()
            H.append(h_t)
        H = torch.stack(H, dim=0)
        H = self.activation(H)
        # for the batchnorm, we should have (num_nodes, conv_dim, seq_len)
        if self.use_layer_norm:
            H = self.bn(H)
        else:
            H = torch.movedim(H, (0, 1), (2, 0))
            H = self.bn(H)
            H = torch.movedim(H, (0, 1), (1, 2))
        H = self.dropout(H)
        if self.linear is not None:
            H = self.linear(H)
        return H
    
    @staticmethod
    def switch_activation(act):
        if act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'relu':
            return nn.ReLU()
        else:
            raise ValueError
    
class Decoder(nn.Module):
    """
    Decoder module for a Convolutional LSTM Autoencoder.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layer output from the encoder.
    num_channels : int
        Number of channels in the output tensor.
    K_cheb : int
        Chebyshev polynomial degree.
    device : torch.device, optional
        Device to run the module on (default: 'cuda' if available, else 'cpu').
    activation : str, optional
        Activation function to use (default: 'tanh').
    dropout : float, optional
        Dropout probability (default: 0.5).
    """
    def __init__(self,
                 conv_dim,
                 num_channels,
                 K_cheb=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 activation='tanh',
                 dropout=0.5,
                 linear_dim: int=None,
                 use_layer_norm=False,
                ):
        super(Decoder, self).__init__()
        self.conv_dim     = conv_dim
        self.num_channels = num_channels
        self.K_cheb       = K_cheb
        self.device       = device
        self.use_layer_norm = use_layer_norm
        
        if linear_dim is not None:
            self.linear_dim = linear_dim
            # self.linear = nn.Linear(conv_dim, num_channels)
            hidden_dim = (conv_dim + num_channels) // 2
            self.linear = nn.Sequential(
                nn.Linear(conv_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_channels)
            )
            self.rnn_cell = GConvLSTM(
                in_channels=linear_dim,
                out_channels=conv_dim,
                K=self.K_cheb,
            )
        else:
            self.linear = None
            self.rnn_cell = GConvLSTM(
                in_channels=conv_dim,
                out_channels=num_channels,
                K=self.K_cheb,
            )

        self.activation = self.switch_activation(activation)
        if use_layer_norm:
            self.bn         = nn.LayerNorm(conv_dim)
        else:
            self.bn         = nn.BatchNorm1d(conv_dim)
        self.dropout    = nn.Dropout(dropout)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, H, edge_index):
        """
        Forward pass of the Decoder module.

        Parameters
        ----------
        H : torch.Tensor
            Input tensor of shape (seq_len, num_nodes, conv_dim) or 
            (seq_len, num_nodes, linear_dim) from the encoder.
        edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (num_nodes, num_channels, seq_len).
        """
        seq_len, num_nodes, _ = H.shape
        
        # reshape H to (num_nodes, conv_dim, seq_len)
        H = torch.movedim(H, (0, 1), (2, 0))

        # Decode the sequence
        X = []
        h, c = None, None
        # for every t in seq_len, we will apply a ConvLSTM to H[..., t], which
        # is of shape (num_nodes, conv_dim); the RNN will transform it into
        # something of shape (num_nodes, num_channels)
        for t in range(seq_len):
            h, c = self.rnn_cell(H[..., t], edge_index, None, H=h, C=c)
            X.append(h)

        # Finally we stack this seq_len tensors to have something of shape
        # (num_nodes, conv_dim, seq_len)
        X = torch.stack(X, dim=2)
        # activation, batchnorm, dropout
        X = self.activation(X)
        if self.use_layer_norm:
            X = torch.movedim(X, 1, 2)
            X = self.bn(X)
            X = torch.movedim(X, 1, 2)
        else:
            X = self.bn(X)
        X = self.dropout(X)
        if self.linear is not None:
            X = torch.movedim(X, 1, 2)
            X = self.linear(X)
            X = torch.movedim(X, 1, 2)
        
        # Apply sigmoid, as we know our data is in the range [0, 1]
        X = self.sigmoid(X)

        return X
    
    @staticmethod
    def switch_activation(act):
        if act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'relu':
            return nn.ReLU()
        else:
            raise ValueError
    
class Autoencoder(nn.Module):
    """
    Autoencoder neural network model.

    Parameters
    ----------
    conv_dim : int
        Dimension of the output of the encoder.
    linear_dim : int
        Dimension of the latent space; if None, no linear layer is used and
        the dimension is simply conv_dim.
    num_channels : int
        Number of input channels.
    K_cheb : int, optional
        Number of Chebyshev polynomial basis functions. Defaults to 1.
    device : torch.device, optional
        Device to run the model on. Defaults to 'cuda' if available, else 'cpu'.
    activation : str, optional
        Activation function to use. Defaults to 'tanh'.
    dropout : float, optional
        Dropout probability. Defaults to 0.5.
    """
    def __init__(self,
                 conv_dim,
                 num_channels,
                 K_cheb=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 activation='tanh',
                 dropout=0.5,
                 linear_dim: int=None,
                 use_layer_norm=False,
                ):
        super(Autoencoder, self).__init__()
        self.conv_dim     = conv_dim
        self.num_channels = num_channels
        self.K_cheb       = K_cheb
        self.device       = device
        self.activation   = activation
        self.dropout      = dropout
        self.linear_dim   = linear_dim
        self.use_layer_norm = use_layer_norm
    
        self.encoder = Encoder(
                 conv_dim=conv_dim,
                 num_channels=num_channels,
                 K_cheb=K_cheb,
                 device=device,
                 activation=activation,
                 dropout=dropout,
                 linear_dim=linear_dim,
                 use_layer_norm=use_layer_norm,
        )
        self.decoder = Decoder(
                 conv_dim=conv_dim,
                 num_channels=num_channels,
                 K_cheb=K_cheb,
                 device=device,
                 activation=activation,
                 dropout=dropout,
                 linear_dim=linear_dim,
                 use_layer_norm=use_layer_norm,
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (num_nodes, num_channels, -1).
        edge_index : torch.Tensor
            Graph edge indices.

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor.
        """
        H = self.encoder(x, edge_index)
        X = self.decoder(H, edge_index)
        return X
    
    def get_specs(self,):
        return f'{self.conv_dim=}_{self.num_channels=}_{self.K_cheb=}_\
            {self.activation=}_{self.linear_dim=}'