# Quick Start: Model Compression untuk YOLO

Panduan cepat untuk menggunakan fitur model compression yang telah ditambahkan.

## 🚀 Quick Examples

### 1. L1 Regularization (Sparsity Training)
```bash
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00001 structured_l1=True
```

### 2. Channel Pruning
```bash
yolo train model=yolo26n.pt data=coco8.yaml epochs=50 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=10
```

### 3. Knowledge Distillation
```bash
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    distillation=True teacher_model=yolo26x.pt temperature=4.0 distill_alpha=0.7
```

### 4. Kombinasi Lengkap
```bash
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00005 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=50 \
    distillation=True teacher_model=yolo26x.pt temperature=4.0
```

## 📊 Analisis Model

```bash
# Analyze sparsity
python compression_utils.py analyze yolo26n.pt

# Compare models
python compression_utils.py compare yolo26n.pt yolo26n_compressed.pt

# Generate detailed report
python compression_utils.py report yolo26n.pt --output report.txt

# Get pruning suggestions
python compression_utils.py suggest yolo26n.pt --target 0.5
```

## 📝 Workflow Recommended

### Workflow 3-Stage (Best Results)

```bash
# Stage 1: Sparsity Training (100 epoch)
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00005 structured_l1=True lambda_bn=0.0001 \
    name=stage1_sparsity

# Stage 2: Pruning (50 epoch)
yolo train model=runs/detect/stage1_sparsity/weights/best.pt data=coco8.yaml epochs=50 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=10 importance_metric=bn \
    name=stage2_pruned

# Stage 3: Fine-tune dengan Distillation (50 epoch)
yolo train model=runs/detect/stage2_pruned/weights/best.pt data=coco8.yaml epochs=50 \
    distillation=True teacher_model=yolo26x.pt temperature=4.0 distill_alpha=0.7 \
    name=stage3_final
```

### Workflow 1-Stage (Quick)

```bash
# All-in-one training
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00005 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=50 \
    distillation=True teacher_model=yolo26x.pt temperature=4.0 distill_alpha=0.7 \
    name=compressed_model
```

## 🐍 Python API

```python
from ultralytics import YOLO

# Training dengan compression
model = YOLO('yolo26n.yaml')
model.train(
    data='coco8.yaml',
    epochs=100,
    l1_regularization=True,
    lambda_l1=1e-5,
    pruning=True,
    pruning_ratio=0.3,
    prune_at_epoch=50,
    distillation=True,
    teacher_model='yolo26x.pt',
    temperature=4.0,
    distill_alpha=0.7
)

# Validation
results = model.val()
print(f"mAP50-95: {results.box.map}")
```

## 📚 Parameter Reference

### L1 Regularization
- `l1_regularization`: Enable L1 (default: False)
- `lambda_l1`: L1 coefficient (default: 1e-5, range: 1e-6 to 1e-4)
- `structured_l1`: Channel-level sparsity (default: False)
- `lambda_bn`: BatchNorm L1 coefficient (default: 1e-4)

### Pruning
- `pruning`: Enable pruning (default: False)
- `pruning_ratio`: Fraction to prune (default: 0.3, range: 0.1-0.5)
- `prune_at_epoch`: When to prune (default: 50)
- `pruning_method`: 'channel' or 'layer' (default: 'channel')
- `importance_metric`: 'l1', 'l2', or 'bn' (default: 'l1')

### Knowledge Distillation
- `distillation`: Enable distillation (default: False)
- `teacher_model`: Path to teacher model (required)
- `temperature`: Softening temperature (default: 4.0, range: 1.0-10.0)
- `distill_alpha`: Distillation weight (default: 0.7, range: 0.0-1.0)
- `distill_features`: Distill features too (default: False)
- `distill_type`: 'response', 'feature', or 'yolo' (default: 'response')

## 💡 Tips

### L1 Regularization
- Start with `lambda_l1=1e-5`
- Use `structured_l1=True` for channel pruning
- Train for 50-100 epochs minimum
- Monitor sparsity dengan `compression_utils.py analyze`

### Pruning
- Do sparsity training first
- Prune at middle of training (e.g., epoch 50/100)
- `pruning_ratio=0.3` adalah safe starting point
- Use `importance_metric=bn` dengan structured L1

### Distillation
- Teacher harus 2x lebih besar dari student
- `temperature=4.0` adalah optimal untuk YOLO
- `distill_alpha=0.7` balances distillation & hard targets
- Lebih banyak epoch = better results

## 🔍 Monitoring

Check training logs untuk:
- Sparsity progression
- Pruning statistics
- Distillation loss
- Validation mAP

Gunakan `compression_utils.py` untuk analisis mendalam.

## 📖 Full Documentation

Lihat [MODEL_COMPRESSION.md](MODEL_COMPRESSION.md) untuk dokumentasi lengkap.

## 🙋 Examples

Lihat `examples/model_compression_examples.py` untuk contoh lengkap dengan kode Python.

---

**Need help?** Check dokumentasi lengkap atau buka issue di repository.
