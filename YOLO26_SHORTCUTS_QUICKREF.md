# YOLO26 Shortcut-Aware Pruning - Quick Reference

## Quick Start

```python
from ultralytics import YOLO

# Training with automatic shortcut-aware pruning
model = YOLO('yolo26.yaml')
model.train(
    data='coco.yaml',
    pruning=True,
    pruning_ratio=0.3,
    prune_at_epoch=50
)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Auto-Detection** | Automatically detects YOLO26 architecture |
| **Shortcut Preservation** | Preserves residual, CSP, and fusion shortcuts |
| **Adaptive Pruning** | Different ratios for different layer types |
| **Configuration** | YAML-based shortcut definitions |

## Shortcut Types & Pruning Ratios

Given base pruning_ratio = 0.3:

| Shortcut Type | Multiplier | Effective Ratio | Example Layers |
|---------------|------------|-----------------|----------------|
| Residual | 0.5 | 15% | Bottleneck with add |
| CSP Shortcut | 0.7 | 21% | C3k2, C2f |
| Feature Fusion | 0.6 | 18% | Concat layers |
| Attention | 0.75 | 22.5% | C2PSA |
| Regular | 1.0 | 30% | Standard Conv |

## YOLO26 Architecture Map

### Backbone
```
Layer 0-1:   Conv + Conv
Layer 2:     C3k2 (shortcut=False)
Layer 3-4:   Conv + C3k2 (shortcut=False)
Layer 5-6:   Conv + C3k2 (shortcut=True) ✓
Layer 7-8:   Conv + C3k2 (shortcut=True) ✓
Layer 9:     SPPF ✓
Layer 10:    C2PSA ✓
```

### Head
```
Layer 11-12: Upsample + Concat [11,6] ✓
Layer 13:    C3k2 (shortcut=True) ✓

Layer 14-15: Upsample + Concat [14,4] ✓
Layer 16:    C3k2 (shortcut=True) ✓ P3/8

Layer 17-18: Conv + Concat [17,13] ✓
Layer 19:    C3k2 (shortcut=True) ✓ P4/16

Layer 20-21: Conv + Concat [20,10] ✓
Layer 22:    C3k2 (shortcut=True) ✓ P5/32

Layer 23:    Detect
```

✓ = Has shortcuts (conservative pruning applied)

## Common Commands

### Training with Pruning
```bash
yolo train model=yolo26.yaml data=coco.yaml pruning=True pruning_ratio=0.3 prune_at_epoch=50
```

### Manual Pruning
```python
from ultralytics.utils.pruning import ChannelPruner
pruner = ChannelPruner(model, pruning_ratio=0.3, config_file='yolo26')
masks = pruner.prune()
```

### Check Shortcuts
```python
config = load_shortcut_config('yolo26')
print(f"Backbone shortcuts: {len(config['backbone'])}")
print(f"Head shortcuts: {len(config['head'])}")
```

## Configuration File Location

```
ultralytics/cfg/pruning/yolo26_shortcuts.yaml
```

## Files Reference

| File | Purpose |
|------|---------|
| `pruning.py` | Core pruning logic |
| `yolo26_shortcuts.yaml` | Shortcut configuration |
| `yolo26_pruning_shortcuts.py` | Usage examples |
| `trainer.py` | Training integration |

## Parameters

### ChannelPruner
```python
ChannelPruner(
    model,                      # Model to prune
    pruning_ratio=0.3,          # Base pruning ratio
    importance_metric='l1',     # 'l1', 'l2', or 'bn'
    preserve_shortcuts=True,    # Enable shortcut preservation
    config_file='yolo26'        # Config name
)
```

### Training Args
```python
model.train(
    pruning=True,               # Enable pruning
    pruning_ratio=0.3,          # Base ratio
    prune_at_epoch=50,          # When to prune
    pruning_method='channel',   # Method
    importance_metric='l1',     # Metric
    l1_regularization=True,     # Prepare for pruning
    structured_l1=True          # Channel-level L1
)
```

## Customization

### Modify Pruning Multipliers
Edit `yolo26_shortcuts.yaml`:
```yaml
pruning_strategy:
  residual:
    pruning_ratio_multiplier: 0.4  # More aggressive
  csp_shortcut:
    pruning_ratio_multiplier: 0.8  # Less conservative
```

### Create Custom Config
1. Copy `yolo26_shortcuts.yaml`
2. Rename to `custom_shortcuts.yaml`
3. Modify layer definitions
4. Use: `config_file='custom'`

## Testing

```bash
python test_pruning_shortcuts.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dimension mismatch | Enable `preserve_shortcuts=True` |
| Poor performance | Reduce `pruning_ratio` |
| Config not found | Check file path and name |
| Import error | Install PyYAML: `pip install pyyaml` |

## Best Practices

✅ DO:
- Use shortcut preservation for YOLO26
- Start with low pruning ratios (0.1-0.2)
- Fine-tune after pruning
- Use structured L1 regularization

❌ DON'T:
- Disable shortcut preservation
- Prune too aggressively initially
- Skip post-pruning validation

## Performance Expectations

| Metric | Typical Result |
|--------|----------------|
| Size Reduction | 15-25% |
| Accuracy Drop | 0.5-2.0 mAP |
| Speedup | 1.2-1.5x |
| Training Stability | High |

## Links

- Main Documentation: `YOLO26_PRUNING_SHORTCUTS.md`
- Indonesian Documentation: `YOLO26_PRUNING_SHORTCUTS_ID.md`
- Examples: `examples/yolo26_pruning_shortcuts.py`
- Configuration: `ultralytics/cfg/pruning/yolo26_shortcuts.yaml`

## Support

For issues and questions:
- GitHub Issues: https://github.com/ultralytics/ultralytics/issues
- Documentation: https://docs.ultralytics.com

---

**Last Updated**: 2026
**Version**: 1.0
**License**: Ultralytics AGPL-3.0
