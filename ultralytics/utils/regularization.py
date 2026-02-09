# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
L1 Regularization for Sparsity Training in YOLO models.

This module implements L1 regularization techniques to promote sparsity in neural networks,
which is useful for model compression through pruning.

Usage:
    from ultralytics.utils.regularization import L1Regularizer
    
    regularizer = L1Regularizer(lambda_l1=1e-5)
    l1_loss = regularizer(model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.utils import LOGGER


class L1Regularizer:
    """
    L1 Regularization for promoting sparsity in neural network weights.
    
    L1 regularization adds the sum of absolute values of all parameters to the loss,
    encouraging many weights to become exactly zero, which facilitates pruning.
    
    Attributes:
        lambda_l1 (float): L1 regularization coefficient. Higher values promote more sparsity.
        target_layers (tuple): Types of layers to apply L1 regularization to.
        
    Example:
        >>> regularizer = L1Regularizer(lambda_l1=1e-5)
        >>> total_loss = base_loss + regularizer(model)
    """
    
    def __init__(self, lambda_l1=1e-5, target_layers=(nn.Conv2d, nn.BatchNorm2d)):
        """
        Initialize L1Regularizer.
        
        Args:
            lambda_l1 (float): L1 regularization coefficient (default: 1e-5).
            target_layers (tuple): Layer types to apply regularization to.
        """
        self.lambda_l1 = lambda_l1
        self.target_layers = target_layers
        LOGGER.info(f"L1 Regularizer initialized with lambda={lambda_l1}")
    
    def __call__(self, model):
        """
        Calculate L1 regularization loss for the model.
        
        Args:
            model (nn.Module): The neural network model.
            
        Returns:
            torch.Tensor: L1 regularization loss.
        """
        l1_loss = 0.0
        for module in model.modules():
            if isinstance(module, self.target_layers):
                for param in module.parameters():
                    if param.requires_grad:
                        l1_loss += torch.sum(torch.abs(param))
        
        return self.lambda_l1 * l1_loss
    
    def get_sparsity(self, model, threshold=1e-3):
        """
        Calculate current sparsity of the model.
        
        Args:
            model (nn.Module): The neural network model.
            threshold (float): Threshold below which weights are considered zero.
            
        Returns:
            dict: Dictionary containing sparsity statistics.
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, self.target_layers):
                layer_zeros = 0
                layer_total = 0
                for param in module.parameters():
                    if param.requires_grad:
                        layer_total += param.numel()
                        layer_zeros += (torch.abs(param) < threshold).sum().item()
                
                if layer_total > 0:
                    layer_sparsity[name] = layer_zeros / layer_total
                    total_params += layer_total
                    zero_params += layer_zeros
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            'overall_sparsity': overall_sparsity,
            'total_params': total_params,
            'zero_params': zero_params,
            'layer_sparsity': layer_sparsity
        }


class StructuredL1Regularizer(L1Regularizer):
    """
    Structured L1 Regularization for channel-level sparsity.
    
    This regularizer promotes sparsity at the channel/filter level rather than individual weights,
    making it more suitable for structured pruning.
    
    Example:
        >>> regularizer = StructuredL1Regularizer(lambda_l1=1e-4)
        >>> total_loss = base_loss + regularizer(model)
    """
    
    def __init__(self, lambda_l1=1e-4, target_layers=(nn.Conv2d,)):
        """Initialize Structured L1 Regularizer for channel-level sparsity."""
        super().__init__(lambda_l1, target_layers)
        LOGGER.info(f"Structured L1 Regularizer initialized with lambda={lambda_l1}")
    
    def __call__(self, model):
        """
        Calculate structured L1 regularization loss (channel-wise).
        
        Args:
            model (nn.Module): The neural network model.
            
        Returns:
            torch.Tensor: Structured L1 regularization loss.
        """
        l1_loss = 0.0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Apply L1 on channel-wise L2 norms
                weight = module.weight
                # weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                channel_norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                l1_loss += torch.sum(torch.abs(channel_norm))
            elif isinstance(module, nn.BatchNorm2d):
                # Apply L1 on BatchNorm scaling factors (gamma)
                if hasattr(module, 'weight') and module.weight is not None:
                    l1_loss += torch.sum(torch.abs(module.weight))
        
        return self.lambda_l1 * l1_loss
    
    def get_channel_importance(self, model):
        """
        Get importance scores for each channel in convolutional layers.
        
        Args:
            model (nn.Module): The neural network model.
            
        Returns:
            dict: Dictionary mapping layer names to channel importance scores.
        """
        channel_importance = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                # Calculate L2 norm for each output channel
                importance = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                channel_importance[name] = importance.detach().cpu()
            elif isinstance(module, nn.BatchNorm2d):
                # Use BatchNorm scaling factors as importance
                if hasattr(module, 'weight') and module.weight is not None:
                    channel_importance[name] = torch.abs(module.weight.detach().cpu())
        
        return channel_importance


def get_bn_l1_loss(model, lambda_bn=1e-4):
    """
    Calculate L1 loss on BatchNorm scaling factors only.
    
    This is a common approach for channel pruning, where we penalize
    the BatchNorm scaling factors (gamma) to zero.
    
    Args:
        model (nn.Module): The neural network model.
        lambda_bn (float): BatchNorm L1 regularization coefficient.
        
    Returns:
        torch.Tensor: L1 loss on BatchNorm scaling factors.
    """
    bn_loss = 0.0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                bn_loss += torch.sum(torch.abs(module.weight))
    
    return lambda_bn * bn_loss
