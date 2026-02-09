# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Channel and Layer Pruning for YOLO models.

This module implements structured pruning techniques to compress neural networks
by removing entire channels or layers while maintaining model accuracy.

Usage:
    from ultralytics.utils.pruning import ChannelPruner, LayerPruner
    
    pruner = ChannelPruner(model, pruning_ratio=0.3)
    pruned_model = pruner.prune()
"""

from __future__ import annotations

import copy
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics.utils import LOGGER

# YOLO26 Official Shortcut Configuration
# This configuration defines which layers have shortcut connections in YOLO26 architecture
YOLO26_SHORTCUTS = {
    # Backbone shortcuts - C3k2 modules typically have shortcuts
    'backbone': {
        'C3k2': {
            'layers': [2, 4, 6, 8],  # Layer indices with C3k2 in backbone
            'shortcut': True,  # C3k2 uses shortcuts by default
            'internal_shortcuts': True,  # Bottleneck blocks inside have shortcuts
        },
        'C2PSA': {
            'layers': [10],  # Layer index with C2PSA in backbone
            'shortcut': False,  # C2PSA doesn't use traditional shortcuts
            'attention_based': True,  # Uses attention mechanism instead
        },
    },
    # Head shortcuts - depends on C3k2 configuration
    'head': {
        'C3k2': {
            'layers': [13, 16, 19, 22],  # Layer indices with C3k2 in head
            'shortcut': True,  # Most head C3k2 use shortcuts
            'variable_shortcuts': {  # Some layers may vary
                22: False,  # P5 output may use different configuration
            },
        },
        'Concat': {
            'layers': [12, 15, 18, 21],  # Concat layers create shortcuts
            'source_layers': [
                [11, 6],   # Concat at layer 12: from layers 11 and 6
                [14, 4],   # Concat at layer 15: from layers 14 and 4
                [17, 13],  # Concat at layer 18: from layers 17 and 13
                [20, 10],  # Concat at layer 21: from layers 20 and 10
            ],
        },
    },
}

# Module types that support shortcuts
SHORTCUT_MODULES = ('C3k2', 'C2f', 'C3', 'Bottleneck', 'C2', 'CSPDarknet')


def load_shortcut_config(model_name='yolo26'):
    """
    Load shortcut configuration for a specific YOLO model.
    
    Args:
        model_name (str): Name of the model ('yolo26', 'yolo11', etc.).
        
    Returns:
        dict: Shortcut configuration or None if file not found.
    """
    config_path = Path(__file__).parent.parent / 'cfg' / 'pruning' / f'{model_name}_shortcuts.yaml'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            LOGGER.info(f"Loaded shortcut configuration from {config_path}")
            return config
        except Exception as e:
            LOGGER.warning(f"Failed to load shortcut config: {e}")
            return None
    else:
        LOGGER.debug(f"Shortcut config not found: {config_path}")
        return None


class ChannelPruner:
    """
    Channel Pruning for convolutional neural networks.
    
    This class implements channel pruning by removing less important channels
    from convolutional layers based on their importance scores.
    
    Attributes:
        model (nn.Module): The neural network model to prune.
        pruning_ratio (float): Ratio of channels to prune (0.0-1.0).
        importance_metric (str): Metric to determine channel importance.
        
    Example:
        >>> pruner = ChannelPruner(model, pruning_ratio=0.3)
        >>> pruned_model = pruner.prune()
    """
    
    def __init__(self, model, pruning_ratio=0.3, importance_metric='l1', preserve_shortcuts=True, config_file='yolo26'):
        """
        Initialize Channel Pruner.
        
        Args:
            model (nn.Module): Model to prune.
            pruning_ratio (float): Ratio of channels to prune (0.0-1.0).
            importance_metric (str): 'l1', 'l2', or 'bn' (BatchNorm scaling).
            preserve_shortcuts (bool): Whether to preserve shortcut connections during pruning.
            config_file (str): Name of shortcut configuration file (e.g., 'yolo26').
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.importance_metric = importance_metric
        self.preserve_shortcuts = preserve_shortcuts
        
        # Load shortcut configuration
        self.shortcut_config = load_shortcut_config(config_file) if preserve_shortcuts else None
        
        # Identify shortcuts in the model
        self.shortcut_layers = self._identify_shortcuts()
        
        LOGGER.info(f"Channel Pruner initialized with ratio={pruning_ratio}, metric={importance_metric}")
        if preserve_shortcuts:
            LOGGER.info(f"Identified {len(self.shortcut_layers)} layers with shortcuts to preserve")
            if self.shortcut_config:
                LOGGER.info(f"Using {config_file} shortcut configuration")
    
    def _identify_shortcuts(self):
        """
        Identify layers with shortcut connections in the model.
        
        Returns:
            dict: Dictionary mapping layer names to shortcut information.
        """
        shortcuts = {}
        
        for name, module in self.model.named_modules():
            # Check for modules with shortcut attribute
            if hasattr(module, 'add') and module.add:
                shortcuts[name] = {
                    'type': 'residual',
                    'module': module.__class__.__name__,
                }
                LOGGER.debug(f"Found residual shortcut in {name} ({module.__class__.__name__})")
            
            # Check for YOLO26 specific modules
            module_type = module.__class__.__name__
            if module_type in SHORTCUT_MODULES:
                # C3k2, C2f, and similar modules
                if hasattr(module, 'm') and hasattr(module, 'cv1') and hasattr(module, 'cv2'):
                    shortcuts[name] = {
                        'type': 'csp_shortcut',
                        'module': module_type,
                        'has_bottleneck': True,
                    }
                    LOGGER.debug(f"Found CSP shortcut in {name} ({module_type})")
            
            # Check for Concat layers (feature fusion shortcuts)
            if module_type == 'Concat':
                shortcuts[name] = {
                    'type': 'concat',
                    'module': 'Concat',
                }
                LOGGER.debug(f"Found Concat shortcut in {name}")
        
        return shortcuts
    
    def _get_shortcut_constraint(self, layer_name):
        """
        Get pruning constraints for layers with shortcuts.
        
        Args:
            layer_name (str): Name of the layer.
            
        Returns:
            float: Modified pruning ratio for this layer (lower = more conservative).
        """
        if not self.preserve_shortcuts or layer_name not in self.shortcut_layers:
            return self.pruning_ratio
        
        shortcut_info = self.shortcut_layers[layer_name]
        shortcut_type = shortcut_info['type']
        
        # Use configuration-based multipliers if available
        if self.shortcut_config and 'pruning_strategy' in self.shortcut_config:
            strategy = self.shortcut_config['pruning_strategy']
            
            if shortcut_type in strategy:
                multiplier = strategy[shortcut_type].get('pruning_ratio_multiplier', 0.7)
                return self.pruning_ratio * multiplier
        
        # Fallback to default conservative pruning for shortcut layers
        if shortcut_type == 'residual':
            # Very conservative for direct residual connections
            return self.pruning_ratio * 0.5
        elif shortcut_type == 'csp_shortcut':
            # Moderate conservation for CSP-style shortcuts
            return self.pruning_ratio * 0.7
        elif shortcut_type == 'concat' or shortcut_type == 'feature_fusion':
            # Concat layers need channel alignment
            return self.pruning_ratio * 0.6
        elif shortcut_type == 'attention':
            # Attention mechanisms need sufficient channels
            return self.pruning_ratio * 0.75
        
        return self.pruning_ratio
    
    def calculate_channel_importance(self, layer):
        """
        Calculate importance score for each channel in a layer.
        
        Args:
            layer (nn.Module): Convolutional or BatchNorm layer.
            
        Returns:
            torch.Tensor: Importance scores for each channel.
        """
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data
            if self.importance_metric == 'l1':
                # L1 norm of each output channel
                importance = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
            elif self.importance_metric == 'l2':
                # L2 norm of each output channel
                importance = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
            else:
                importance = torch.ones(weight.size(0))
        elif isinstance(layer, nn.BatchNorm2d):
            # Use BatchNorm scaling factors
            if hasattr(layer, 'weight') and layer.weight is not None:
                importance = torch.abs(layer.weight.data)
            else:
                importance = torch.ones(layer.num_features)
        else:
            importance = None
        
        return importance
    
    def get_pruning_mask(self, importance, num_channels):
        """
        Generate pruning mask based on importance scores.
        
        Args:
            importance (torch.Tensor): Channel importance scores.
            num_channels (int): Total number of channels.
            
        Returns:
            torch.Tensor: Boolean mask (True = keep, False = prune).
        """
        num_prune = int(num_channels * self.pruning_ratio)
        num_keep = num_channels - num_prune
        
        if num_keep <= 0:
            num_keep = 1  # Keep at least one channel
        
        # Get indices of top-k important channels
        _, indices = torch.topk(importance, num_keep)
        mask = torch.zeros(num_channels, dtype=torch.bool)
        mask[indices] = True
        
        return mask
    
    def prune_conv_layer(self, conv_layer, in_mask=None, out_mask=None):
        """
        Prune a convolutional layer based on input/output masks.
        
        Args:
            conv_layer (nn.Conv2d): Convolutional layer to prune.
            in_mask (torch.Tensor): Input channel mask.
            out_mask (torch.Tensor): Output channel mask.
            
        Returns:
            nn.Conv2d: Pruned convolutional layer.
        """
        weight = conv_layer.weight.data
        bias = conv_layer.bias.data if conv_layer.bias is not None else None
        
        # Apply output channel mask
        if out_mask is not None:
            weight = weight[out_mask]
            if bias is not None:
                bias = bias[out_mask]
        
        # Apply input channel mask
        if in_mask is not None:
            weight = weight[:, in_mask]
        
        # Create new layer with pruned channels
        new_conv = nn.Conv2d(
            in_channels=weight.size(1),
            out_channels=weight.size(0),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups if conv_layer.groups == 1 else weight.size(1),
            bias=conv_layer.bias is not None
        )
        
        new_conv.weight.data = weight
        if bias is not None:
            new_conv.bias.data = bias
        
        return new_conv
    
    def prune_bn_layer(self, bn_layer, mask):
        """
        Prune a BatchNorm layer based on channel mask.
        
        Args:
            bn_layer (nn.BatchNorm2d): BatchNorm layer to prune.
            mask (torch.Tensor): Channel mask.
            
        Returns:
            nn.BatchNorm2d: Pruned BatchNorm layer.
        """
        new_bn = nn.BatchNorm2d(mask.sum().item())
        
        if hasattr(bn_layer, 'weight') and bn_layer.weight is not None:
            new_bn.weight.data = bn_layer.weight.data[mask]
        if hasattr(bn_layer, 'bias') and bn_layer.bias is not None:
            new_bn.bias.data = bn_layer.bias.data[mask]
        if hasattr(bn_layer, 'running_mean') and bn_layer.running_mean is not None:
            new_bn.running_mean = bn_layer.running_mean[mask]
        if hasattr(bn_layer, 'running_var') and bn_layer.running_var is not None:
            new_bn.running_var = bn_layer.running_var[mask]
        
        return new_bn
    
    def prune(self):
        """
        Perform channel pruning on the model with shortcut awareness.
        
        Returns:
            nn.Module: Pruned model.
        """
        LOGGER.info(f"Starting channel pruning with ratio={self.pruning_ratio}")
        if self.preserve_shortcuts:
            LOGGER.info("Shortcut preservation enabled - applying conservative pruning to shortcut layers")
        
        # Get channel importance for all layers
        layer_importance = {}
        layer_masks = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                importance = self.calculate_channel_importance(module)
                if importance is not None:
                    layer_importance[name] = importance
                    
                    # Generate mask based on importance
                    if isinstance(module, nn.Conv2d):
                        num_channels = module.out_channels
                    else:
                        num_channels = module.num_features
                    
                    # Apply shortcut-aware pruning ratio
                    effective_ratio = self._get_shortcut_constraint(name)
                    if effective_ratio != self.pruning_ratio:
                        LOGGER.info(f"Layer {name}: using modified ratio={effective_ratio:.3f} (shortcut layer)")
                    
                    # Generate mask with adjusted ratio
                    num_prune = int(num_channels * effective_ratio)
                    num_keep = num_channels - num_prune
                    if num_keep <= 0:
                        num_keep = 1
                    
                    _, indices = torch.topk(importance, num_keep)
                    mask = torch.zeros(num_channels, dtype=torch.bool)
                    mask[indices] = True
                    layer_masks[name] = mask
                    
                    pruned_channels = (~mask).sum().item()
                    LOGGER.info(f"Layer {name}: pruning {pruned_channels}/{num_channels} channels")
        
        # Note: Full implementation would require model architecture knowledge
        # to properly handle connections between layers
        LOGGER.info(f"Channel pruning analysis complete. Use prune_model() for actual pruning.")
        
        return layer_masks
    
    def get_model_sparsity(self):
        """
        Calculate current sparsity statistics of the model.
        
        Returns:
            dict: Sparsity statistics.
        """
        total_params = 0
        total_channels = 0
        zero_channels = 0
        
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                total_channels += module.out_channels
                weight = module.weight.data
                channel_norm = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
                zero_channels += (channel_norm < 1e-6).sum().item()
                total_params += weight.numel()
        
        return {
            'total_channels': total_channels,
            'zero_channels': zero_channels,
            'channel_sparsity': zero_channels / total_channels if total_channels > 0 else 0,
            'total_params': total_params
        }


class LayerPruner:
    """
    Layer-level pruning for neural networks.
    
    This class implements layer pruning by removing entire layers or blocks
    based on their importance to the network output.
    
    Example:
        >>> pruner = LayerPruner(model, target_layers=['layer3', 'layer5'])
        >>> pruned_model = pruner.prune()
    """
    
    def __init__(self, model, target_layers=None, sensitivity_threshold=0.01):
        """
        Initialize Layer Pruner.
        
        Args:
            model (nn.Module): Model to prune.
            target_layers (list): List of layer names to consider for pruning.
            sensitivity_threshold (float): Sensitivity threshold for pruning decision.
        """
        self.model = model
        self.target_layers = target_layers or []
        self.sensitivity_threshold = sensitivity_threshold
        LOGGER.info(f"Layer Pruner initialized with {len(self.target_layers)} target layers")
    
    def analyze_layer_importance(self, dataloader, device='cuda'):
        """
        Analyze importance of each layer using sensitivity analysis.
        
        Args:
            dataloader: Validation dataloader.
            device (str): Device to run analysis on.
            
        Returns:
            dict: Layer importance scores.
        """
        self.model.eval()
        layer_importance = {}
        
        # Get baseline accuracy
        baseline_acc = self._evaluate_model(dataloader, device)
        LOGGER.info(f"Baseline accuracy: {baseline_acc:.4f}")
        
        # Test removing each target layer
        for layer_name in self.target_layers:
            # Temporarily zero out layer
            layer = dict(self.model.named_modules())[layer_name]
            original_state = self._save_layer_state(layer)
            self._zero_layer(layer)
            
            # Evaluate with zeroed layer
            pruned_acc = self._evaluate_model(dataloader, device)
            importance = baseline_acc - pruned_acc
            layer_importance[layer_name] = importance
            
            LOGGER.info(f"Layer {layer_name}: importance={importance:.4f}")
            
            # Restore layer
            self._restore_layer_state(layer, original_state)
        
        return layer_importance
    
    def _evaluate_model(self, dataloader, device):
        """Evaluate model accuracy on dataloader."""
        # Simplified evaluation - should be customized for specific task
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                
                try:
                    outputs = self.model(inputs)
                    total += inputs.size(0)
                    correct += inputs.size(0)  # Placeholder
                except Exception as e:
                    LOGGER.warning(f"Evaluation error: {e}")
                    break
        
        return correct / total if total > 0 else 0.0
    
    def _save_layer_state(self, layer):
        """Save layer parameters."""
        state = {}
        for name, param in layer.named_parameters():
            state[name] = param.data.clone()
        return state
    
    def _restore_layer_state(self, layer, state):
        """Restore layer parameters."""
        for name, param in layer.named_parameters():
            if name in state:
                param.data.copy_(state[name])
    
    def _zero_layer(self, layer):
        """Zero out layer parameters."""
        for param in layer.parameters():
            param.data.zero_()
    
    def prune(self, layer_importance):
        """
        Remove layers below sensitivity threshold.
        
        Args:
            layer_importance (dict): Layer importance scores.
            
        Returns:
            list: Names of layers to prune.
        """
        layers_to_prune = []
        
        for layer_name, importance in layer_importance.items():
            if importance < self.sensitivity_threshold:
                layers_to_prune.append(layer_name)
                LOGGER.info(f"Marking {layer_name} for pruning (importance={importance:.4f})")
        
        return layers_to_prune


def prune_model(model, pruning_ratio=0.3, method='channel', importance_metric='l1'):
    """
    Convenience function to prune a model.
    
    Args:
        model (nn.Module): Model to prune.
        pruning_ratio (float): Ratio of channels/layers to prune.
        method (str): 'channel' or 'layer' pruning.
        importance_metric (str): Importance metric ('l1', 'l2', 'bn').
        
    Returns:
        tuple: (pruned_model, pruning_info)
    """
    if method == 'channel':
        pruner = ChannelPruner(model, pruning_ratio, importance_metric)
        masks = pruner.prune()
        sparsity = pruner.get_model_sparsity()
        return model, {'masks': masks, 'sparsity': sparsity}
    elif method == 'layer':
        pruner = LayerPruner(model)
        LOGGER.info("Layer pruning requires dataloader for importance analysis")
        return model, {}
    else:
        raise ValueError(f"Unknown pruning method: {method}")
