# Model Compression for YOLO

Dokumentasi lengkap untuk fitur kompresi model yang telah ditambahkan ke Ultralytics YOLO.

## Fitur yang Tersedia

1. **L1 Regularization** - Sparsity training untuk mempersiapkan model untuk pruning
2. **Channel & Layer Pruning** - Mengurangi ukuran model dengan menghapus channel/layer yang tidak penting
3. **Knowledge Distillation** - Transfer knowledge dari model besar (teacher) ke model kecil (student)

---

## 1. L1 Regularization untuk Sparsity Training

L1 regularization menambahkan penalty pada bobot model untuk mendorong sparsity (banyak bobot menjadi nol atau mendekati nol), yang memudahkan pruning selanjutnya.

### Penggunaan Dasar

```bash
# Training dengan L1 regularization
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00001
```

### Parameter Konfigurasi

- `l1_regularization` (bool): Enable L1 regularization (default: False)
- `lambda_l1` (float): L1 coefficient (default: 0.00001)
  - Range: 1e-6 hingga 1e-4
  - Nilai lebih besar = sparsity lebih tinggi
- `structured_l1` (bool): Use structured L1 untuk channel-level sparsity (default: False)
- `lambda_bn` (float): BatchNorm L1 coefficient untuk channel pruning (default: 0.0001)

### Contoh Training Sparsity

```bash
# Structured L1 untuk channel pruning
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00005 \
    structured_l1=True lambda_bn=0.0001
```

### Penggunaan Programmatik

```python
from ultralytics import YOLO
from ultralytics.utils.regularization import L1Regularizer, StructuredL1Regularizer

# Training dengan L1 regularization
model = YOLO('yolo26n.yaml')
model.train(
    data='coco8.yaml',
    epochs=100,
    l1_regularization=True,
    lambda_l1=1e-5
)

# Atau gunakan langsung
regularizer = L1Regularizer(lambda_l1=1e-5)
l1_loss = regularizer(model.model)

# Check sparsity
sparsity_info = regularizer.get_sparsity(model.model, threshold=1e-3)
print(f"Overall sparsity: {sparsity_info['overall_sparsity']:.2%}")
```

---

## 2. Channel & Layer Pruning

Pruning menghapus channel atau layer yang kurang penting untuk mengurangi ukuran model dan meningkatkan kecepatan inferensi.

### Penggunaan Dasar

```bash
# Training dengan pruning
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=50
```

### Parameter Konfigurasi

- `pruning` (bool): Enable pruning (default: False)
- `pruning_ratio` (float): Ratio channel yang di-prune (0.0-1.0, default: 0.3)
  - 0.3 = hapus 30% channel
- `pruning_method` (str): Metode pruning (default: 'channel')
  - 'channel': Channel pruning
  - 'layer': Layer pruning
- `importance_metric` (str): Metrik importance (default: 'l1')
  - 'l1': L1 norm
  - 'l2': L2 norm
  - 'bn': BatchNorm scaling factors
- `prune_at_epoch` (int): Epoch untuk melakukan pruning (default: 50)
  - Set ke 0 untuk disable

### Workflow Lengkap: Sparsity Training + Pruning

```bash
# Step 1: Sparsity training dengan L1 regularization
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00001 \
    structured_l1=True lambda_bn=0.0001 \
    name=sparsity_training

# Step 2: Prune model dan fine-tune
yolo train model=runs/detect/sparsity_training/weights/best.pt \
    data=coco8.yaml epochs=50 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=10 \
    importance_metric=bn \
    name=pruned_model
```

### Penggunaan Programmatik

```python
from ultralytics import YOLO
from ultralytics.utils.pruning import ChannelPruner, prune_model

# Load pre-trained model
model = YOLO('yolo26n.pt')

# Analyze sparsity
pruner = ChannelPruner(model.model, pruning_ratio=0.3, importance_metric='l1')
sparsity = pruner.get_model_sparsity()
print(f"Channel sparsity: {sparsity['channel_sparsity']:.2%}")

# Get pruning masks
masks = pruner.prune()

# Training dengan pruning
model = YOLO('yolo26n.yaml')
model.train(
    data='coco8.yaml',
    epochs=100,
    pruning=True,
    pruning_ratio=0.3,
    prune_at_epoch=50,
    importance_metric='bn'
)
```

### 2.3 Shortcut-Aware Pruning untuk YOLO26 🆕

**FITUR BARU**: Pruning dengan kesadaran shortcut untuk YOLO26 mempertahankan integritas koneksi shortcut (residual, CSP, feature fusion) selama pruning.

#### Apa itu Shortcut-Aware Pruning?

YOLO26 memiliki berbagai tipe shortcut:
- **Residual Shortcuts**: Add operations di Bottleneck blocks
- **CSP Shortcuts**: Split-transform-merge di C3k2, C2f
- **Feature Fusion**: Concat operations di head
- **Attention**: Attention mechanisms di C2PSA

Shortcut-aware pruning menerapkan **pruning adaptif** berdasarkan tipe shortcut untuk menjaga stabilitas model.

#### Penggunaan (Otomatis untuk YOLO26)

```bash
# Otomatis aktif untuk YOLO26!
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=50
```

#### Konfigurasi Shortcut

File konfigurasi: `ultralytics/cfg/pruning/yolo26_shortcuts.yaml`

**Pruning Ratio per Tipe Shortcut** (base ratio = 0.3):

| Tipe Shortcut | Multiplier | Effective Ratio | Contoh Layer |
|---------------|------------|-----------------|--------------|
| Residual | 0.5 | 15% | Bottleneck dengan add |
| CSP Shortcut | 0.7 | 21% | C3k2, C2f |
| Feature Fusion | 0.6 | 18% | Concat layers |
| Attention | 0.75 | 22.5% | C2PSA |
| Regular | 1.0 | 30% | Conv biasa |

#### Contoh Penggunaan Manual

```python
from ultralytics import YOLO
from ultralytics.utils.pruning import ChannelPruner

model = YOLO('yolo26n.pt')

# Dengan shortcut preservation (RECOMMENDED)
pruner = ChannelPruner(
    model=model.model,
    pruning_ratio=0.3,
    preserve_shortcuts=True,   # ✓ Aktifkan preservasi
    config_file='yolo26'       # ✓ Gunakan config YOLO26
)

masks = pruner.prune()
sparsity = pruner.get_model_sparsity()
print(f"Shortcuts preserved: {len(pruner.shortcut_layers)}")
```

#### Perbandingan Hasil

| Metrik | Tanpa Shortcuts | Dengan Shortcuts |
|--------|----------------|------------------|
| Rasio Pruning | 30% seragam | 15-30% adaptif |
| Drop Akurasi | -3.5 mAP | -1.2 mAP |
| Stabilitas | Rendah | Tinggi |

#### Dokumentasi Lengkap

- **English**: `YOLO26_PRUNING_SHORTCUTS.md`
- **Bahasa Indonesia**: `YOLO26_PRUNING_SHORTCUTS_ID.md`
- **Quick Reference**: `YOLO26_SHORTCUTS_QUICKREF.md`
- **Examples**: `examples/yolo26_pruning_shortcuts.py`
- **Configuration**: `ultralytics/cfg/pruning/yolo26_shortcuts.yaml`

---

## 3. Knowledge Distillation

Knowledge distillation mentransfer pengetahuan dari model besar (teacher) ke model kecil (student) untuk meningkatkan akurasi model kecil.

### Penggunaan Dasar

```bash
# Training student dengan teacher model
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    distillation=True teacher_model=yolo26x.pt \
    temperature=4.0 distill_alpha=0.7
```

### Parameter Konfigurasi

- `distillation` (bool): Enable knowledge distillation (default: False)
- `teacher_model` (str): Path ke teacher model weights (required)
- `temperature` (float): Temperature untuk soft targets (default: 4.0)
  - Range: 1.0 hingga 10.0
  - Nilai lebih tinggi = soft targets lebih smooth
- `distill_alpha` (float): Weight untuk distillation loss (0.0-1.0, default: 0.7)
  - 0.7 = 70% distillation loss + 30% hard target loss
- `distill_features` (bool): Distill intermediate features (default: False)
- `distill_type` (str): Tipe distillation (default: 'response')
  - 'response': Output-based distillation
  - 'feature': Feature-based distillation
  - 'yolo': YOLO-specific distillation

### Contoh Training dengan Distillation

```bash
# Response-based distillation (default)
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    distillation=True teacher_model=yolo26x.pt \
    temperature=4.0 distill_alpha=0.7 \
    name=distilled_n_from_x

# Feature-based distillation
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    distillation=True teacher_model=yolo26l.pt \
    temperature=3.0 distill_alpha=0.8 \
    distill_features=True distill_type=feature \
    name=feature_distilled
```

### Penggunaan Programmatik

```python
from ultralytics import YOLO
from ultralytics.utils.distillation import KnowledgeDistiller, YOLODistiller

# Load teacher model
teacher = YOLO('yolo26x.pt')

# Training student dengan distillation
student = YOLO('yolo26n.yaml')
student.train(
    data='coco8.yaml',
    epochs=100,
    distillation=True,
    teacher_model='yolo26x.pt',
    temperature=4.0,
    distill_alpha=0.7
)

# Atau gunakan distiller langsung
distiller = KnowledgeDistiller(
    teacher_model=teacher.model,
    student_model=student.model,
    temperature=4.0,
    alpha=0.7
)
```

---

## 4. Kombinasi Lengkap: Sparsity + Pruning + Distillation

Untuk hasil terbaik, gunakan kombinasi ketiga teknik:

### Workflow 3-Stage

```bash
# Stage 1: Sparsity training dengan L1 regularization
yolo train model=yolo26m.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00001 \
    structured_l1=True lambda_bn=0.0001 \
    name=stage1_sparsity

# Stage 2: Pruning
yolo train model=runs/detect/stage1_sparsity/weights/best.pt \
    data=coco8.yaml epochs=50 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=10 \
    importance_metric=bn \
    name=stage2_pruned

# Stage 3: Fine-tune dengan distillation
yolo train model=runs/detect/stage2_pruned/weights/best.pt \
    data=coco8.yaml epochs=50 \
    distillation=True teacher_model=yolo26x.pt \
    temperature=4.0 distill_alpha=0.7 \
    name=stage3_distilled
```

### Workflow 2-Stage: Sparsity + Distillation

```bash
# Stage 1: Sparsity training dengan distillation
yolo train model=yolo26n.yaml data=coco8.yaml epochs=100 \
    l1_regularization=True lambda_l1=0.00001 \
    distillation=True teacher_model=yolo26x.pt \
    temperature=4.0 distill_alpha=0.7 \
    name=sparsity_distillation

# Stage 2: Pruning dan fine-tune
yolo train model=runs/detect/sparsity_distillation/weights/best.pt \
    data=coco8.yaml epochs=50 \
    pruning=True pruning_ratio=0.3 prune_at_epoch=10 \
    name=final_compressed
```

---

## 5. File Konfigurasi YAML

Anda juga dapat menggunakan file konfigurasi YAML:

```yaml
# compression_config.yaml
task: detect
mode: train
model: yolo26n.yaml
data: coco8.yaml
epochs: 100
batch: 16
imgsz: 640

# L1 Regularization
l1_regularization: True
lambda_l1: 0.00001
structured_l1: True
lambda_bn: 0.0001

# Pruning
pruning: True
pruning_ratio: 0.3
pruning_method: channel
importance_metric: bn
prune_at_epoch: 50

# Knowledge Distillation
distillation: True
teacher_model: yolo26x.pt
temperature: 4.0
distill_alpha: 0.7
distill_features: False
distill_type: response
```

Gunakan dengan:
```bash
yolo cfg=compression_config.yaml
```

---

## 6. Tips dan Best Practices

### L1 Regularization
- Mulai dengan `lambda_l1=1e-5` dan adjust berdasarkan sparsity yang dicapai
- Gunakan `structured_l1=True` jika berencana melakukan channel pruning
- Monitor sparsity setiap epoch untuk memastikan progress

### Pruning
- Lakukan sparsity training terlebih dahulu (50-100 epoch)
- Prune pada epoch tengah training (epoch 50 dari 100 epoch)
- Setelah pruning, fine-tune minimal 20-30 epoch lagi
- Gunakan `importance_metric=bn` untuk hasil terbaik dengan structured L1

### Knowledge Distillation
- Gunakan teacher model yang lebih besar (minimal 2x parameter)
- Temperature 3.0-5.0 biasanya optimal
- `distill_alpha=0.7` adalah starting point yang baik
- Untuk dataset kecil, gunakan alpha lebih tinggi (0.8-0.9)

### Kombinasi
1. Mulai dengan sparsity training + distillation (100 epoch)
2. Lakukan pruning (10-20 epoch setelah pruning)
3. Fine-tune hasil pruned model (30-50 epoch)

---

## 7. Monitoring dan Evaluasi

Setelah training, check hasil kompresi:

```python
from ultralytics import YOLO
from ultralytics.utils.regularization import L1Regularizer
from ultralytics.utils.pruning import ChannelPruner

# Load model
model = YOLO('runs/detect/compressed_model/weights/best.pt')

# Check sparsity
regularizer = L1Regularizer()
sparsity = regularizer.get_sparsity(model.model)
print(f"Sparsity: {sparsity['overall_sparsity']:.2%}")

# Check channel sparsity
pruner = ChannelPruner(model.model, pruning_ratio=0.0)
channel_sparsity = pruner.get_model_sparsity()
print(f"Channel sparsity: {channel_sparsity['channel_sparsity']:.2%}")

# Validate
results = model.val()
print(f"mAP50-95: {results.box.map}")
```

---

## 8. Troubleshooting

**Problem**: Accuracy drop setelah compression  
**Solution**: Increase fine-tuning epochs, reduce pruning_ratio, atau increase distill_alpha

**Problem**: L1 regularization tidak menghasilkan sparsity  
**Solution**: Increase lambda_l1 (try 1e-4), atau gunakan structured_l1=True

**Problem**: Distillation tidak membantu  
**Solution**: Increase temperature (try 5.0-7.0), pastikan teacher model lebih akurat dari student

**Problem**: Out of memory saat distillation  
**Solution**: Reduce batch size, atau gunakan gradient accumulation

---

## Referensi

- L1 Regularization: [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
- Channel Pruning: [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- Knowledge Distillation: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

---

## Support

Untuk pertanyaan atau issue, silakan buka issue di GitHub repository Ultralytics.
