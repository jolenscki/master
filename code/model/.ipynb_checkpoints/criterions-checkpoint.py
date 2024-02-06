import torch
import torch.nn as nn
import math

# Weighted Mean Squared Error Loss
class WeightedMSELoss(nn.Module):
    def __init__(self, weight=10):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        mask = target != 0
        loss = (input - target) ** 2
        loss[mask] *= self.weight
        return loss.mean()

# Mean Squared Logarithmic Error Loss
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.epsilon = 1e-08

    def forward(self, input, target):
        first_log = torch.log(input + self.epsilon + 1.)
        second_log = torch.log(target + self.epsilon + 1.)
        return torch.mean((first_log - second_log) ** 2)

# Focal Loss for Regression
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        loss = torch.abs(input - target)
        loss = self.alpha * (loss ** self.gamma)
        return loss.mean()

# Zero-Inflated Loss
class ZeroInflatedLoss(nn.Module):
    def __init__(self, non_zero_weight=10, zero_weight=0.1):
        super(ZeroInflatedLoss, self).__init__()
        self.non_zero_weight = non_zero_weight
        self.zero_weight = zero_weight

    def forward(self, input, target):
        non_zero_mask = target != 0
        zero_mask = ~non_zero_mask
        loss = (input - target) ** 2
        loss[non_zero_mask] *= self.non_zero_weight
        loss[zero_mask] *= self.zero_weight
        return loss.mean()

# Custom Huber Loss with Zero Inflation
class CustomHuberLoss(nn.Module):
    def __init__(self, delta=1.0, zero_inflation_weight=0.5):
        super(CustomHuberLoss, self).__init__()
        self.delta = delta
        self.zero_inflation_weight = zero_inflation_weight

    def forward(self, input, target):
        huber_loss = torch.where(torch.abs(input - target) < self.delta, 
                                 0.5 * ((input - target) ** 2),
                                 self.delta * (torch.abs(input - target) - 0.5 * self.delta))
        zero_mask = (target == 0)
        huber_loss[zero_mask] *= self.zero_inflation_weight
        return huber_loss.mean()
    
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Calculates the log-cosh loss between the predicted and true values.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_true : torch.Tensor
            The true values.

        Returns
        -------
        torch.Tensor
            The log-cosh loss.
        """
        def _log_cosh(x):
            return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
        return torch.mean(_log_cosh(y_pred - y_true))