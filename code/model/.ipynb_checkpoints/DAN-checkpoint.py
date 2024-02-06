from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TopKPooling

class GradientReversalLayer(nn.Module):
    """
    A PyTorch module that applies gradient reversal to its input.

    Parameters
    ----------
    lambda_ : float, optional
        The scaling factor for the reversed gradient. Default is 1.0.

    Returns
    -------
    torch.Tensor
        The input tensor with reversed gradients.
    """
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GradientReversalFunction(torch.autograd.Function):
    """
    Implements the gradient reversal layer used in domain adaptation tasks.
    This function multiplies the gradient of the loss function with a negative
    scalar value lambda_.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    lambda_ : float
        Scalar value to multiply the gradient of the loss function with.
    
    Returns
    -------
    torch.Tensor
        The input tensor `x`.
    
    Examples
    --------
    >>> x = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    >>> lambda_ = 0.5
    >>> y = GradientReversalFunction.apply(x, lambda_)
    >>> y.backward(torch.ones_like(y))
    >>> x.grad
    tensor([-0.5000, -0.5000, -0.5000])
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = -ctx.lambda_ * grad_output
        return grad_input, None
    
class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        seq_len, 
        feat_dim,
        left_nodes=250,
        dropout_rate=0.5,
        use_layer_norm=False
    ):
        super(DomainDiscriminator, self).__init__()
        
        self.feat_dim = feat_dim
        self.left_nodes = left_nodes
        self.pool = TopKPooling(feat_dim, ratio=left_nodes)
        
        lin_input = feat_dim * left_nodes
        self.linear = nn.Sequential(
            nn.Linear(lin_input, lin_input//2),
            nn.ReLU(),
            nn.Linear(lin_input//2, 1)
        )
        
        if use_layer_norm:
            self.bn         = nn.LayerNorm(lin_input)
        else:
            self.bn         = nn.BatchNorm1d(lin_input)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(
        self, 
        features,
        edge_index,
        batch, 
        city, 
        tgt,
        return_logits=False
    ):
        '''
        Parameters
        ----------
        features: torch.Tensor
            of shape (seq_len, num_nodes, features)
        edge_index: torch.Tensor
            of shape (2, num_edges)
        batch: torch.Tensor
            of shape (num_nodes, )
        city: str
            name of the current batch
        tgt: str
            name of the target domain
        return_logits: bool, optional
            whether to return the logits

        Returns
        -------
        Union[torch.Tensor, float]:
            returns a tensor with the probabilities if return_logits is True
            returns the BCE loss if return_logits is False
        '''
        x_ = features[-1, ...]
        # shape of x: (left_nodes, feat_dim)        
        x_, edge_index_, _, batch_, _, _ = self.pool(x_, edge_index, batch=batch)
        
        batch_size = batch.unique().size(0)
        x_ = x_.view(batch_size, -1)
        
        x_ = self.bn(x_)
        x_ = self.dropout(x_)

        # shape of domain_logits: (batch_size, 1)
        domain_logits = self.linear(x_).to(torch.float32)

        if city == tgt:
            batch_truth = torch.zeros(batch_size, device=features.device)
        else:
            batch_truth = torch.ones(batch_size, device=features.device)

        # Using binary_cross_entropy_with_logits
        loss = F.binary_cross_entropy_with_logits(
            domain_logits, 
            batch_truth.unsqueeze(1).to(torch.float32)
        )
        
        if return_logits:
            return domain_logits

        return loss