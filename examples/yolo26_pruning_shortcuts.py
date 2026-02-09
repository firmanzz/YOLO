"""
YOLO26 Pruning with Shortcut Preservation - Usage Examples
===========================================================

This file demonstrates how to use channel and layer pruning with official YOLO26
shortcut configurations to ensure model integrity during compression.

Author: Ultralytics
Date: 2026
"""

from ultralytics import YOLO
from ultralytics.utils.pruning import ChannelPruner, load_shortcut_config


def example_1_basic_pruning_with_shortcuts():
    """
    Example 1: Basic channel pruning with automatic shortcut preservation.
    
    This example shows the simplest way to prune a YOLO26 model while
    automatically preserving shortcut connections.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Pruning with Shortcut Preservation")
    print("="*70)
    
    # Load YOLO26 model
    model = YOLO('yolo26n.pt')
    
    # Create pruner with shortcut preservation enabled (default)
    pruner = ChannelPruner(
        model=model.model,
        pruning_ratio=0.3,           # Prune 30% of channels
        importance_metric='l1',       # Use L1 norm for importance
        preserve_shortcuts=True,      # Enable shortcut preservation
        config_file='yolo26'          # Use YOLO26 shortcut config
    )
    
    # Perform pruning analysis
    masks = pruner.prune()
    
    print(f"\nPruning complete!")
    print(f"Generated masks for {len(masks)} layers")
    
    # Get sparsity statistics
    sparsity = pruner.get_model_sparsity()
    print(f"\nModel Sparsity Statistics:")
    print(f"  Total channels: {sparsity['total_channels']}")
    print(f"  Zero channels: {sparsity['zero_channels']}")
    print(f"  Channel sparsity: {sparsity['channel_sparsity']:.2%}")


def example_2_training_with_pruning():
    """
    Example 2: Training YOLO26 with pruning at specific epoch.
    
    This demonstrates how to integrate pruning into the training workflow,
    automatically applying shortcut-aware pruning at a specified epoch.
    """
    print("\n" + "="*70)
    print("Example 2: Training with Pruning")
    print("="*70)
    
    # Load YOLO26 model
    model = YOLO('yolo26.yaml')  # or use pretrained: 'yolo26n.pt'
    
    # Train with pruning enabled
    # The trainer will automatically use YOLO26 shortcut configuration
    results = model.train(
        data='coco.yaml',
        epochs=100,
        imgsz=640,
        
        # Pruning configuration
        pruning=True,              # Enable pruning
        pruning_ratio=0.3,         # Prune 30% of channels
        prune_at_epoch=50,         # Apply pruning at epoch 50
        pruning_method='channel',  # Channel-level pruning
        importance_metric='l1',    # L1 importance metric
        
        # Optional: L1 regularization to prepare model for pruning
        l1_regularization=True,
        lambda_l1=1e-5,
        structured_l1=True,        # Channel-level L1 for better pruning
    )
    
    print("\nTraining complete with pruning!")


def example_3_custom_shortcut_config():
    """
    Example 3: Loading and inspecting the YOLO26 shortcut configuration.
    
    This shows how to access the shortcut configuration to understand
    which layers have shortcuts and how they're treated during pruning.
    """
    print("\n" + "="*70)
    print("Example 3: Inspecting Shortcut Configuration")
    print("="*70)
    
    # Load the YOLO26 shortcut configuration
    config = load_shortcut_config('yolo26')
    
    if config:
        print("\nYOLO26 Shortcut Configuration:")
        print(f"Model: {config.get('model_name', 'Unknown')}")
        print(f"Version: {config.get('architecture_version', 'Unknown')}")
        
        # Display backbone shortcuts
        print("\n--- Backbone Shortcuts ---")
        if 'backbone' in config:
            for module_type, info in config['backbone'].items():
                print(f"\n{module_type}:")
                if 'layers' in info:
                    for layer_info in info['layers']:
                        print(f"  Layer {layer_info.get('index')}: "
                              f"shortcut={layer_info.get('shortcut', 'N/A')}")
        
        # Display head shortcuts
        print("\n--- Head Shortcuts ---")
        if 'head' in config:
            for module_type, info in config['head'].items():
                print(f"\n{module_type}:")
                if 'layers' in info:
                    for layer_info in info['layers']:
                        if module_type == 'Concat':
                            print(f"  Layer {layer_info.get('index')}: "
                                  f"sources={layer_info.get('sources', [])}")
                        else:
                            print(f"  Layer {layer_info.get('index')}: "
                                  f"shortcut={layer_info.get('shortcut', 'N/A')}")
        
        # Display pruning strategies
        print("\n--- Pruning Strategies ---")
        if 'pruning_strategy' in config:
            for shortcut_type, strategy in config['pruning_strategy'].items():
                multiplier = strategy.get('pruning_ratio_multiplier', 1.0)
                min_ch = strategy.get('min_channels', 'N/A')
                print(f"{shortcut_type}:")
                print(f"  Ratio multiplier: {multiplier}")
                print(f"  Min channels: {min_ch}")


def example_4_progressive_pruning():
    """
    Example 4: Progressive pruning with shortcut preservation.
    
    This demonstrates iterative pruning with increasing ratios,
    ensuring shortcuts are preserved at each stage.
    """
    print("\n" + "="*70)
    print("Example 4: Progressive Pruning")
    print("="*70)
    
    # Load YOLO26 model
    model = YOLO('yolo26n.pt')
    
    # Progressive pruning ratios
    pruning_ratios = [0.1, 0.2, 0.3]
    
    for ratio in pruning_ratios:
        print(f"\n--- Pruning with ratio {ratio} ---")
        
        # Create pruner with current ratio
        pruner = ChannelPruner(
            model=model.model,
            pruning_ratio=ratio,
            importance_metric='l1',
            preserve_shortcuts=True,
            config_file='yolo26'
        )
        
        # Analyze pruning
        masks = pruner.prune()
        sparsity = pruner.get_model_sparsity()
        
        print(f"Channel sparsity: {sparsity['channel_sparsity']:.2%}")
        print(f"Identified shortcuts: {len(pruner.shortcut_layers)}")
        
        # In practice, you would fine-tune the model here before next iteration
        # model.train(data='coco.yaml', epochs=10, ...)


def example_5_compare_with_without_shortcuts():
    """
    Example 5: Compare pruning with and without shortcut preservation.
    
    This demonstrates the difference in pruning behavior when
    shortcut preservation is enabled vs disabled.
    """
    print("\n" + "="*70)
    print("Example 5: Pruning Comparison")
    print("="*70)
    
    # Load YOLO26 model
    model = YOLO('yolo26n.pt')
    
    # Pruning WITHOUT shortcut preservation
    print("\n--- WITHOUT Shortcut Preservation ---")
    pruner_no_shortcuts = ChannelPruner(
        model=model.model,
        pruning_ratio=0.3,
        importance_metric='l1',
        preserve_shortcuts=False  # Disabled
    )
    masks_no_shortcuts = pruner_no_shortcuts.prune()
    sparsity_no_shortcuts = pruner_no_shortcuts.get_model_sparsity()
    
    print(f"Shortcuts identified: {len(pruner_no_shortcuts.shortcut_layers)}")
    print(f"Channel sparsity: {sparsity_no_shortcuts['channel_sparsity']:.2%}")
    
    # Reload model for fair comparison
    model = YOLO('yolo26n.pt')
    
    # Pruning WITH shortcut preservation
    print("\n--- WITH Shortcut Preservation ---")
    pruner_with_shortcuts = ChannelPruner(
        model=model.model,
        pruning_ratio=0.3,
        importance_metric='l1',
        preserve_shortcuts=True,  # Enabled
        config_file='yolo26'
    )
    masks_with_shortcuts = pruner_with_shortcuts.prune()
    sparsity_with_shortcuts = pruner_with_shortcuts.get_model_sparsity()
    
    print(f"Shortcuts identified: {len(pruner_with_shortcuts.shortcut_layers)}")
    print(f"Channel sparsity: {sparsity_with_shortcuts['channel_sparsity']:.2%}")
    
    print("\n--- Comparison ---")
    print(f"Difference in identified shortcuts: "
          f"{len(pruner_with_shortcuts.shortcut_layers) - len(pruner_no_shortcuts.shortcut_layers)}")
    print(f"Sparsity difference: "
          f"{abs(sparsity_with_shortcuts['channel_sparsity'] - sparsity_no_shortcuts['channel_sparsity']):.2%}")


def example_6_layer_specific_analysis():
    """
    Example 6: Analyze specific layers with shortcuts.
    
    This shows how to inspect which layers have shortcuts and
    how they're being treated during pruning.
    """
    print("\n" + "="*70)
    print("Example 6: Layer-Specific Analysis")
    print("="*70)
    
    # Load YOLO26 model
    model = YOLO('yolo26n.pt')
    
    # Create pruner
    pruner = ChannelPruner(
        model=model.model,
        pruning_ratio=0.3,
        importance_metric='l1',
        preserve_shortcuts=True,
        config_file='yolo26'
    )
    
    # Display identified shortcuts
    print(f"\nIdentified {len(pruner.shortcut_layers)} layers with shortcuts:\n")
    
    for layer_name, shortcut_info in pruner.shortcut_layers.items():
        shortcut_type = shortcut_info['type']
        module_type = shortcut_info['module']
        
        # Get effective pruning ratio for this layer
        effective_ratio = pruner._get_shortcut_constraint(layer_name)
        reduction = (1 - effective_ratio / pruner.pruning_ratio) * 100
        
        print(f"{layer_name}:")
        print(f"  Module: {module_type}")
        print(f"  Shortcut Type: {shortcut_type}")
        print(f"  Base Pruning Ratio: {pruner.pruning_ratio:.2%}")
        print(f"  Effective Ratio: {effective_ratio:.2%}")
        print(f"  Reduction: {reduction:.1f}%")
        print()


if __name__ == '__main__':
    """
    Run all examples to demonstrate YOLO26 pruning with shortcut preservation.
    """
    print("\n" + "="*70)
    print("YOLO26 Pruning with Shortcut Preservation - Examples")
    print("="*70)
    
    # Note: Uncomment the examples you want to run
    # Some examples require model files to be present
    
    # example_1_basic_pruning_with_shortcuts()
    # example_2_training_with_pruning()
    example_3_custom_shortcut_config()
    # example_4_progressive_pruning()
    # example_5_compare_with_without_shortcuts()
    # example_6_layer_specific_analysis()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
