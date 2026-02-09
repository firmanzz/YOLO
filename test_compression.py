"""
Script sederhana untuk memulai kompresi model YOLO.

Jalankan script ini untuk quick test semua fitur compression.
"""

from ultralytics import YOLO
from ultralytics.utils import LOGGER

def test_l1_regularization():
    """Test L1 regularization."""
    print("\n" + "="*60)
    print("Testing L1 Regularization")
    print("="*60)
    
    model = YOLO('yolo26n.yaml')
    model.train(
        data='coco8.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        l1_regularization=True,
        lambda_l1=1e-5,
        name='test_l1'
    )
    
    print("✓ L1 Regularization test completed")

def test_pruning():
    """Test channel pruning."""
    print("\n" + "="*60)
    print("Testing Channel Pruning")
    print("="*60)
    
    model = YOLO('yolo26n.pt')
    model.train(
        data='coco8.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        pruning=True,
        pruning_ratio=0.3,
        prune_at_epoch=2,
        name='test_pruning'
    )
    
    print("✓ Pruning test completed")

def test_distillation():
    """Test knowledge distillation."""
    print("\n" + "="*60)
    print("Testing Knowledge Distillation")
    print("="*60)
    
    model = YOLO('yolo26n.yaml')
    model.train(
        data='coco8.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        distillation=True,
        teacher_model='yolo26s.pt',  # Use smaller teacher for quick test
        temperature=4.0,
        distill_alpha=0.7,
        name='test_distillation'
    )
    
    print("✓ Distillation test completed")

def test_combined():
    """Test combined compression."""
    print("\n" + "="*60)
    print("Testing Combined Compression")
    print("="*60)
    
    model = YOLO('yolo26n.yaml')
    model.train(
        data='coco8.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        l1_regularization=True,
        lambda_l1=1e-5,
        distillation=True,
        teacher_model='yolo26s.pt',
        temperature=4.0,
        name='test_combined'
    )
    
    print("✓ Combined compression test completed")

def analyze_model():
    """Analyze compressed model."""
    print("\n" + "="*60)
    print("Analyzing Model")
    print("="*60)
    
    from ultralytics.utils.regularization import L1Regularizer
    from ultralytics.utils.pruning import ChannelPruner
    
    try:
        model = YOLO('yolo26n.pt')
        
        # Sparsity analysis
        regularizer = L1Regularizer()
        sparsity = regularizer.get_sparsity(model.model)
        print(f"\nSparsity Analysis:")
        print(f"  Overall sparsity: {sparsity['overall_sparsity']:.2%}")
        print(f"  Total parameters: {sparsity['total_params']:,}")
        
        # Channel analysis
        pruner = ChannelPruner(model.model, pruning_ratio=0.3)
        channel_info = pruner.get_model_sparsity()
        print(f"\nChannel Analysis:")
        print(f"  Total channels: {channel_info['total_channels']}")
        print(f"  Channel sparsity: {channel_info['channel_sparsity']:.2%}")
        
        print("\n✓ Analysis completed")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("YOLO Model Compression - Quick Test")
    print("="*60)
    print("\nThis script will test all compression features.")
    print("Each test runs for only 5 epochs for quick validation.")
    print("\nTests available:")
    print("  1. L1 Regularization")
    print("  2. Channel Pruning")
    print("  3. Knowledge Distillation")
    print("  4. Combined Compression")
    print("  5. Model Analysis")
    print("  6. Run All Tests")
    
    choice = input("\nSelect test to run (1-6): ").strip()
    
    if choice == '1':
        test_l1_regularization()
    elif choice == '2':
        test_pruning()
    elif choice == '3':
        test_distillation()
    elif choice == '4':
        test_combined()
    elif choice == '5':
        analyze_model()
    elif choice == '6':
        print("\nRunning all tests...")
        test_l1_regularization()
        test_pruning()
        test_distillation()
        test_combined()
        analyze_model()
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check results in runs/detect/")
    print("  2. Run: python compression_utils.py analyze <model_path>")
    print("  3. See COMPRESSION_QUICKSTART.md for more examples")
    print("  4. See MODEL_COMPRESSION.md for full documentation")
