#!/usr/bin/env python
"""
Utility script untuk model compression YOLO.

Script ini menyediakan command-line interface untuk:
- Analyze model sparsity
- Perform channel pruning analysis  
- Compare model sizes
- Visualize compression results

Usage:
    python compression_utils.py analyze yolo26n.pt
    python compression_utils.py compare yolo26n.pt yolo26n_compressed.pt
    python compression_utils.py sparsity-report yolo26n.pt
"""

import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def analyze_model_sparsity(model_path, threshold=1e-3):
    """Analyze and report model sparsity."""
    from ultralytics.utils.regularization import L1Regularizer, StructuredL1Regularizer
    from ultralytics.utils.pruning import ChannelPruner
    
    print(f"\n{'='*60}")
    print(f"Model Sparsity Analysis: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # L1 Sparsity Analysis
    print("1. L1 Weight Sparsity Analysis")
    print("-" * 60)
    regularizer = L1Regularizer(lambda_l1=1e-5)
    sparsity = regularizer.get_sparsity(model.model, threshold=threshold)
    
    print(f"Overall Sparsity:        {sparsity['overall_sparsity']:.2%}")
    print(f"Total Parameters:        {sparsity['total_params']:,}")
    print(f"Zero Parameters:         {sparsity['zero_params']:,}")
    print(f"Non-zero Parameters:     {sparsity['total_params'] - sparsity['zero_params']:,}")
    print(f"Threshold:               {threshold}")
    
    # Top sparse layers
    layer_sparsity = sparsity['layer_sparsity']
    if layer_sparsity:
        sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Sparse Layers:")
        for i, (layer_name, sparsity_val) in enumerate(sorted_layers[:10], 1):
            print(f"  {i:2d}. {layer_name:50s} {sparsity_val:.2%}")
    
    # Channel Sparsity Analysis
    print("\n2. Channel-Level Sparsity Analysis")
    print("-" * 60)
    pruner = ChannelPruner(model.model, pruning_ratio=0.0, importance_metric='l1')
    channel_info = pruner.get_model_sparsity()
    
    print(f"Total Channels:          {channel_info['total_channels']}")
    print(f"Zero Channels:           {channel_info['zero_channels']}")
    print(f"Channel Sparsity:        {channel_info['channel_sparsity']:.2%}")
    print(f"Total Parameters:        {channel_info['total_params']:,}")
    
    # Structured L1 Analysis
    print("\n3. Structured L1 Channel Importance")
    print("-" * 60)
    struct_regularizer = StructuredL1Regularizer(lambda_l1=1e-4)
    channel_importance = struct_regularizer.get_channel_importance(model.model)
    
    if channel_importance:
        print(f"Total layers analyzed:   {len(channel_importance)}")
        
        # Show statistics for first few layers
        print("\nChannel Importance Statistics (first 5 layers):")
        for i, (layer_name, importance) in enumerate(list(channel_importance.items())[:5], 1):
            print(f"  {layer_name}")
            print(f"    Mean: {importance.mean():.6f}, Std: {importance.std():.6f}")
            print(f"    Min:  {importance.min():.6f}, Max: {importance.max():.6f}")
    
    print("\n" + "="*60 + "\n")


def compare_models(model1_path, model2_path):
    """Compare two models (size, parameters, accuracy)."""
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}\n")
    
    models = [model1_path, model2_path]
    results = []
    
    for i, model_path in enumerate(models, 1):
        print(f"Analyzing Model {i}: {model_path}")
        
        # Get file size
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            size_mb = 0
            print(f"  Warning: File not found")
        
        # Load and analyze model
        try:
            model = YOLO(model_path)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            # Get sparsity
            from ultralytics.utils.regularization import L1Regularizer
            regularizer = L1Regularizer()
            sparsity = regularizer.get_sparsity(model.model)['overall_sparsity']
            
            results.append({
                'path': model_path,
                'size_mb': size_mb,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'sparsity': sparsity
            })
            
            print(f"  Size:              {size_mb:.2f} MB")
            print(f"  Total Parameters:  {total_params:,}")
            print(f"  Trainable Params:  {trainable_params:,}")
            print(f"  Sparsity:          {sparsity:.2%}")
            print()
            
        except Exception as e:
            print(f"  Error loading model: {e}\n")
            continue
    
    # Comparison
    if len(results) == 2:
        print("Comparison Summary:")
        print("-" * 60)
        
        size_reduction = (1 - results[1]['size_mb'] / results[0]['size_mb']) * 100
        param_reduction = (1 - results[1]['total_params'] / results[0]['total_params']) * 100
        
        print(f"Size Reduction:      {size_reduction:+.1f}%")
        print(f"Parameter Reduction: {param_reduction:+.1f}%")
        
        if results[0]['sparsity'] < results[1]['sparsity']:
            sparsity_increase = (results[1]['sparsity'] - results[0]['sparsity']) * 100
            print(f"Sparsity Increase:   +{sparsity_increase:.1f}%")
        else:
            sparsity_decrease = (results[0]['sparsity'] - results[1]['sparsity']) * 100
            print(f"Sparsity Decrease:   -{sparsity_decrease:.1f}%")
    
    print("\n" + "="*60 + "\n")


def generate_sparsity_report(model_path, output_file=None):
    """Generate detailed sparsity report."""
    from ultralytics.utils.regularization import L1Regularizer, StructuredL1Regularizer
    from ultralytics.utils.pruning import ChannelPruner
    
    print(f"Generating sparsity report for: {model_path}")
    
    model = YOLO(model_path)
    
    # Collect data
    regularizer = L1Regularizer()
    sparsity = regularizer.get_sparsity(model.model, threshold=1e-3)
    
    pruner = ChannelPruner(model.model, pruning_ratio=0.0)
    channel_info = pruner.get_model_sparsity()
    
    struct_regularizer = StructuredL1Regularizer()
    channel_importance = struct_regularizer.get_channel_importance(model.model)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append(f"SPARSITY REPORT: {model_path}")
    report.append("=" * 80)
    report.append("")
    
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Model Path:              {model_path}")
    report.append(f"File Size:               {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    report.append(f"Total Parameters:        {sparsity['total_params']:,}")
    report.append(f"Zero Parameters:         {sparsity['zero_params']:,}")
    report.append(f"Overall Sparsity:        {sparsity['overall_sparsity']:.2%}")
    report.append(f"Total Channels:          {channel_info['total_channels']}")
    report.append(f"Zero Channels:           {channel_info['zero_channels']}")
    report.append(f"Channel Sparsity:        {channel_info['channel_sparsity']:.2%}")
    report.append("")
    
    report.append("LAYER-WISE SPARSITY")
    report.append("-" * 80)
    layer_sparsity = sparsity['layer_sparsity']
    sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
    
    report.append(f"{'Layer Name':<60} {'Sparsity':>10}")
    report.append("-" * 80)
    for layer_name, sparsity_val in sorted_layers:
        report.append(f"{layer_name:<60} {sparsity_val:>9.2%}")
    report.append("")
    
    report.append("CHANNEL IMPORTANCE STATISTICS")
    report.append("-" * 80)
    report.append(f"{'Layer Name':<50} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    report.append("-" * 80)
    for layer_name, importance in channel_importance.items():
        report.append(
            f"{layer_name:<50} "
            f"{importance.mean():>10.6f} "
            f"{importance.std():>10.6f} "
            f"{importance.min():>10.6f} "
            f"{importance.max():>10.6f}"
        )
    
    report.append("")
    report.append("=" * 80)
    
    # Print report
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")


def suggest_pruning_ratio(model_path, target_sparsity=0.5):
    """Suggest pruning ratio based on current sparsity."""
    from ultralytics.utils.regularization import L1Regularizer
    
    print(f"\nAnalyzing model to suggest pruning ratio...")
    print(f"Target sparsity: {target_sparsity:.0%}\n")
    
    model = YOLO(model_path)
    regularizer = L1Regularizer()
    sparsity = regularizer.get_sparsity(model.model)
    
    current_sparsity = sparsity['overall_sparsity']
    print(f"Current sparsity: {current_sparsity:.2%}")
    
    if current_sparsity >= target_sparsity:
        print(f"\nModel already has sparsity >= {target_sparsity:.0%}")
        print(f"Recommended pruning ratio: 0.2-0.3 (conservative)")
    elif current_sparsity >= 0.3:
        ratio = min(0.5, target_sparsity)
        print(f"\nModel has good sparsity (>30%)")
        print(f"Recommended pruning ratio: {ratio:.1f}")
    elif current_sparsity >= 0.1:
        ratio = 0.3
        print(f"\nModel has moderate sparsity")
        print(f"Recommended pruning ratio: {ratio:.1f}")
    else:
        ratio = 0.2
        print(f"\nModel has low sparsity (<10%)")
        print(f"Recommended: Train with L1 regularization first")
        print(f"Then prune with ratio: {ratio:.1f}")
    
    print(f"\nSuggested command:")
    print(f"yolo train model={model_path} data=coco8.yaml epochs=50 "
          f"pruning=True pruning_ratio={ratio:.1f} prune_at_epoch=10")


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Compression Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model sparsity')
    analyze_parser.add_argument('model', type=str, help='Path to model file')
    analyze_parser.add_argument('--threshold', type=float, default=1e-3, 
                               help='Threshold for zero weights (default: 1e-3)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('model1', type=str, help='Path to first model')
    compare_parser.add_argument('model2', type=str, help='Path to second model')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate sparsity report')
    report_parser.add_argument('model', type=str, help='Path to model file')
    report_parser.add_argument('--output', type=str, help='Output file path')
    
    # Suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Suggest pruning ratio')
    suggest_parser.add_argument('model', type=str, help='Path to model file')
    suggest_parser.add_argument('--target', type=float, default=0.5,
                               help='Target sparsity (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_model_sparsity(args.model, args.threshold)
    elif args.command == 'compare':
        compare_models(args.model1, args.model2)
    elif args.command == 'report':
        generate_sparsity_report(args.model, args.output)
    elif args.command == 'suggest':
        suggest_pruning_ratio(args.model, args.target)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
