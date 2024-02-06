from typing import List
from torch_geometric_temporal.nn.recurrent import A3TGCN2
import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self,
                 features: int,
                 linear_dim: int,
                 periods_in: int,
                 periods_out: List[int],
                 num_channels: int,
                 batch_size: int,
                 activation='sigmoid',
                 dropout=0.3,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_layer_norm=False,
                ):
        super(Predictor, self).__init__()
        self.features = features
        self.linear_dim = linear_dim
        self.periods_in = periods_in
        self.periods_out = periods_out
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.device = device
        self.use_layer_norm = use_layer_norm
        
        self.tgnn = A3TGCN2(
            in_channels=self.features,
            out_channels=self.linear_dim,
            periods=self.periods_in,
            batch_size=self.batch_size,
        )
        
        output_dim = len(self.periods_out)*self.num_channels
        # self.linear = nn.Linear(conv_dim, num_channels)
        hidden_dim = (self.linear_dim + output_dim) // 2
        self.linear = nn.Sequential(
            nn.Linear(self.linear_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
        if use_layer_norm:
            self.bn         = nn.LayerNorm(self.linear_dim)
        else:
            self.bn         = nn.BatchNorm1d(self.linear_dim)
        
        self.activation = self.switch_activation(activation)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x, edge_index, batch):
        '''
        Parameters
        ------------
        x: torch.Tensor
            node features, of shape (seq_len, num_nodes*batch_size, features)
        edge_index: torch.Tensor
            edge_indices, of shape (2, batch_size*num_edges)
        batch: torch.Tensor
            batch tensor (tensor that delimits the edge indices of each batch),
            of shape (batch_size*num_edges)
            
        Returns
        ------------
        H: torch.Tensor
            of shape (batch_size, len(self.periods_out), self.num_channels)
        '''

        seq_len, num_nodes, features= x.shape
        x = torch.movedim(x, 0, -1)
        x = x.reshape(self.batch_size, -1, features, seq_len)

        # now we unbatch the edge_index tensor
        edge_index = self.unbatch_edge_index(edge_index, batch)

        H = self.tgnn(X=x, edge_index=edge_index)
        if self.use_layer_norm:
            H = self.bn(H)
        else:
            H = torch.permute(H, (0, 2, 1))
            H = self.bn(H)
            H = torch.permute(H, (0, 2, 1))
        H = self.activation(H)
        H = self.dropout(H)
        
        H = self.linear(H)
        H = H.reshape(-1, self.num_channels, len(self.periods_out))
        H = self.sigmoid(H)

        return H
    
    @staticmethod
    def unbatch_edge_index(edge_index, batch):
        '''
        Static method to unbatch the edge_index tensor.

        Parameters
        ----------
        edge_index: torch.Tensor
            edge_indices, of shape (2, batch_size*num_edges)
        batch: torch.Tensor
            batch tensor (tensor that delimits the edge indices of each batch),
            of shape (batch_size*num_edges)

        Returns
        -------
        edge_subset: torch.Tensor
            edge_indices of the first graph in the batch, of shape (2, num_edges)
        '''
        # Calculate the number of nodes in each graph
        num_nodes_per_graph = torch.bincount(batch)

        # Calculate the cumulative sum of nodes to determine the boundaries
        # between graphs in the edge_index
        # cum_nodes is a tensor that will be like this
        # (N1, N1+N2, ..., N1+N2+...+Nn) with N1 = num_nodes, and most likely
        # N1 = N2 = ... = Nn = N
        # (since we are dealing with batches of graphs of the same city)
        cum_nodes = torch.cumsum(num_nodes_per_graph, dim=0)
        # Add a 0 at the beginning of the tensor
        # now we have (0, N, 2*N, ..., n*N)
        cum_nodes = torch.cat([torch.tensor([0], device=edge_index.device), cum_nodes])

        # Split the edge_index for each graph
        # this mask isolates the edges of the first graph
        # basically edges that are greater than 0 and less than N
        mask = (edge_index[0] >= cum_nodes[0]) & (edge_index[0] < cum_nodes[1])
        edge_subset = edge_index[:, mask]
        # Adjust node indices to start from 0 for each graph
        # this won't do anything on this version (as we are only isolating 
        # first graph, but it will be useful if we need to isolate more graphs)
        edge_subset[0] -= cum_nodes[0]
        edge_subset[1] -= cum_nodes[0]

        return edge_subset
    
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