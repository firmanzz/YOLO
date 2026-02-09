"""
Example script untuk model compression dengan YOLO.

Script ini mendemonstrasikan cara menggunakan L1 regularization,
pruning, dan knowledge distillation untuk kompresi model YOLO.
"""

from ultralytics import YOLO
from ultralytics.utils.regularization import L1Regularizer, StructuredL1Regularizer
from ultralytics.utils.pruning import ChannelPruner
from ultralytics.utils.distillation import KnowledgeDistiller
import torch


def example_l1_regularization():
    """Contoh training dengan L1 regularization untuk sparsity."""
    print("\n=== L1 Regularization Example ===")
    
    # Method 1: Menggunakan parameter training
    model = YOLO('yolo26n.yaml')
    model.train(
        data='coco8.yaml',
        epochs=10,
        batch=16,
        imgsz=640,
        l1_regularization=True,
        lambda_l1=1e-5,
        structured_l1=True,
        lambda_bn=1e-4,
        name='l1_regularization_example'
    )
    
    # Method 2: Manual analysis
    regularizer = L1Regularizer(lambda_l1=1e-5)
    sparsity = regularizer.get_sparsity(model.model, threshold=1e-3)
    print(f"Overall sparsity: {sparsity['overall_sparsity']:.2%}")
    print(f"Total parameters: {sparsity['total_params']:,}")
    print(f"Zero parameters: {sparsity['zero_params']:,}")


def example_sparsity_training():
    """Contoh sparsity training lengkap."""
    print("\n=== Sparsity Training Example ===")
    
    model = YOLO('yolo26n.yaml')
    
    # Train dengan structured L1 untuk channel pruning
    results = model.train(
        data='coco8.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        l1_regularization=True,
        lambda_l1=5e-5,
        structured_l1=True,
        lambda_bn=1e-4,
        name='sparsity_training'
    )
    
    # Check sparsity setelah training
    regularizer = StructuredL1Regularizer(lambda_l1=5e-5)
    channel_importance = regularizer.get_channel_importance(model.model)
    
    print("\nChannel importance scores:")
    for layer_name, importance in list(channel_importance.items())[:5]:
        print(f"  {layer_name}: mean={importance.mean():.6f}, std={importance.std():.6f}")
    
    return model


def example_pruning(model_path='yolo26n.pt'):
    """Contoh channel pruning."""
    print("\n=== Channel Pruning Example ===")
    
    # Load pre-trained atau sparsity-trained model
    model = YOLO(model_path)
    
    # Analyze sparsity sebelum pruning
    pruner = ChannelPruner(model.model, pruning_ratio=0.3, importance_metric='l1')
    sparsity_before = pruner.get_model_sparsity()
    print(f"Sparsity before pruning: {sparsity_before['channel_sparsity']:.2%}")
    
    # Train dengan pruning
    model.train(
        data='coco8.yaml',
        epochs=30,
        batch=16,
        imgsz=640,
        pruning=True,
        pruning_ratio=0.3,
        prune_at_epoch=10,
        importance_metric='bn',
        name='channel_pruning_example'
    )
    
    # Check sparsity setelah pruning
    sparsity_after = pruner.get_model_sparsity()
    print(f"Sparsity after pruning: {sparsity_after['channel_sparsity']:.2%}")
    
    return model


def example_distillation():
    """Contoh knowledge distillation."""
    print("\n=== Knowledge Distillation Example ===")
    
    # Student model (small)
    student = YOLO('yolo26n.yaml')
    
    # Train dengan distillation dari teacher model
    student.train(
        data='coco8.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        distillation=True,
        teacher_model='yolo26m.pt',  # Gunakan model yang lebih besar
        temperature=4.0,
        distill_alpha=0.7,
        distill_type='response',
        name='distillation_example'
    )
    
    # Validate student model
    results = student.val()
    print(f"Student mAP50-95: {results.box.map:.4f}")
    
    return student


def example_full_compression_pipeline():
    """Contoh lengkap: Sparsity + Pruning + Distillation."""
    print("\n=== Full Compression Pipeline Example ===")
    
    # Stage 1: Sparsity Training
    print("\n--- Stage 1: Sparsity Training ---")
    model = YOLO('yolo26n.yaml')
    model.train(
        data='coco8.yaml',
        epochs=30,
        batch=16,
        imgsz=640,
        l1_regularization=True,
        lambda_l1=5e-5,
        structured_l1=True,
        lambda_bn=1e-4,
        name='compression_stage1_sparsity'
    )
    
    # Stage 2: Pruning
    print("\n--- Stage 2: Pruning ---")
    model.train(
        data='coco8.yaml',
        epochs=20,
        batch=16,
        imgsz=640,
        pruning=True,
        pruning_ratio=0.3,
        prune_at_epoch=5,
        importance_metric='bn',
        name='compression_stage2_pruning',
        resume=True
    )
    
    # Stage 3: Fine-tuning dengan Distillation
    print("\n--- Stage 3: Fine-tuning with Distillation ---")
    model.train(
        data='coco8.yaml',
        epochs=30,
        batch=16,
        imgsz=640,
        distillation=True,
        teacher_model='yolo26m.pt',
        temperature=4.0,
        distill_alpha=0.7,
        name='compression_stage3_distillation',
        resume=True
    )
    
    # Final validation
    results = model.val()
    print(f"\nFinal compressed model mAP50-95: {results.box.map:.4f}")
    
    # Model size comparison
    import os
    original_size = os.path.getsize('yolo26n.pt') / (1024 * 1024)
    compressed_size = os.path.getsize('runs/detect/compression_stage3_distillation/weights/best.pt') / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    print(f"Compression ratio: {(1 - compressed_size/original_size)*100:.1f}%")
    
    return model


def example_analyze_model(model_path='yolo26n.pt'):
    """Analyze model sparsity dan channel importance."""
    print("\n=== Model Analysis Example ===")
    
    model = YOLO(model_path)
    
    # L1 Regularizer analysis
    print("\n--- L1 Sparsity Analysis ---")
    regularizer = L1Regularizer(lambda_l1=1e-5)
    sparsity = regularizer.get_sparsity(model.model, threshold=1e-3)
    print(f"Overall sparsity: {sparsity['overall_sparsity']:.2%}")
    print(f"Total parameters: {sparsity['total_params']:,}")
    print(f"Zero parameters: {sparsity['zero_params']:,}")
    
    # Top sparse layers
    layer_sparsity = sparsity['layer_sparsity']
    sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 sparse layers:")
    for layer_name, sparsity_val in sorted_layers[:5]:
        print(f"  {layer_name}: {sparsity_val:.2%}")
    
    # Channel pruner analysis
    print("\n--- Channel Pruning Analysis ---")
    pruner = ChannelPruner(model.model, pruning_ratio=0.3, importance_metric='l1')
    channel_sparsity = pruner.get_model_sparsity()
    print(f"Total channels: {channel_sparsity['total_channels']}")
    print(f"Zero channels: {channel_sparsity['zero_channels']}")
    print(f"Channel sparsity: {channel_sparsity['channel_sparsity']:.2%}")
    
    # Get channel importance
    channel_importance = pruner.calculate_channel_importance
    print("\nChannel importance metric: L1 norm")


def example_combined_training():
    """Contoh training dengan semua fitur sekaligus."""
    print("\n=== Combined Training Example ===")
    
    model = YOLO('yolo26n.yaml')
    
    # Train dengan L1 regularization + Distillation
    model.train(
        data='coco8.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        # L1 Regularization
        l1_regularization=True,
        lambda_l1=5e-5,
        structured_l1=True,
        lambda_bn=1e-4,
        # Pruning (dilakukan di epoch 30)
        pruning=True,
        pruning_ratio=0.3,
        prune_at_epoch=30,
        importance_metric='bn',
        # Knowledge Distillation
        distillation=True,
        teacher_model='yolo26m.pt',
        temperature=4.0,
        distill_alpha=0.7,
        # General
        name='combined_compression'
    )
    
    # Validate hasil
    results = model.val()
    print(f"\nCompressed model mAP50-95: {results.box.map:.4f}")
    
    return model


if __name__ == '__main__':
    # Uncomment contoh yang ingin dijalankan
    
    # Basic examples
    # example_l1_regularization()
    # example_pruning()
    # example_distillation()
    
    # Advanced examples
    # model = example_sparsity_training()
    # example_full_compression_pipeline()
    # example_combined_training()
    
    # Analysis
    # example_analyze_model('yolo26n.pt')
    
    print("\n=== Examples Complete ===")
    print("Uncomment contoh yang ingin dijalankan di bagian __main__")
