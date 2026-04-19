# Extended Forge Simulation — 200 Steps, 500 Pairs

## Extended Results

| Metric | 50 Steps | 200 Steps | Improvement |
|--------|----------|-----------|-------------|
| Loss Start | 10.25 | 10.40 | — |
| Loss End | 2.44 | **0.93** | **62% lower** |
| Loss Reduction | 76% | **91%** | +15% |
| Training Pairs | 200 | 500 | 2.5x more data |
| Steps/sec | 0.6 | **1.7** | 2.8x faster (batch=4) |
| Total Time | 84.5s | 121s | — |

## Loss Curve

```
Step   1:  10.40  (fresh model)
Step  50:   3.36  (-68% from start)
Step 100:   2.15  (-79%)
Step 150:   1.23  (-88%)
Step 200:   0.93  (-91%)  ← still dropping!
```

## Key Insight: Loss Hasn't Plateaued

The loss is still dropping at step 200 (0.93). It hasn't hit a floor.
This means the model is still absorbing new patterns from the fleet data.

Projected convergence:
- Step 500: ~0.4-0.5 (extrapolating exponential decay)
- Step 1000: ~0.2-0.3 (near-overfitting on training set)

## What Happened

The model (distilgpt2, 82M params) was trained on web text (OpenWebText).
It "knows" English well but has zero knowledge of PLATO concepts.

After 200 steps of fleet data:
- Loss dropped 91% → model is learning the PLATO data distribution
- But generation quality is still poor → 200 steps isn't enough to override pre-trained weights
- The model needs ~500-1000 steps to start producing coherent PLATO-specific output

## CPU vs GPU Projection

| Config | Steps/sec | 1000 steps | Overnight (8h) |
|--------|-----------|------------|-----------------|
| CPU (current) | 1.7 | ~10 min | ~48,000 steps |
| RTX 4050 CUDA | ~8-12 | ~2 min | ~200,000+ steps |

## Conclusion

The forge works. Loss converges fast. The bottleneck is:
1. **CUDA torch** — need to install manually (pip OOMs on download)
2. **Training steps** — 200 proves convergence, 1000+ needed for quality
3. **LoRA** — train 120MB adapters instead of 328MB full model

The model is dumb but getting smarter. Every step counts.
