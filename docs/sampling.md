# Direct endpoint interpolation

All three main-task samplers already update the next state by interpolating
between the current state and the model's clean-motion prediction. For a
positive current time `t`, with the time clamp inactive:

```text
x_r = (r / t) * x_t + (1 - r / t) * predicted_x0
    = x_t - (t - r) * ((x_t - predicted_x0) / t)
```

This does not reduce model evaluations: each 1-step/4-step checkpoint retains
its original NFE, time grid and `(t, t-r)` conditioning. A new endpoint prediction
is still made at every step. It is not a shortcut that reuses one prediction for
all four steps.

## What changes

| Task | Main inference path | Change |
| --- | --- | --- |
| T2M | `tasks/t2m/reproduce_main.py` | Install an inference-only copy of the frozen sampler that omits its discarded velocity. |
| S2G | `tasks/s2g/infer.py` | None; the frozen sampler already computes only the direct state update. |
| M2D | Prepared corrected scorer from `tasks/m2d/adapter.py` | Replace one exact sampler assignment, omitting only its discarded velocity. |

T2M and M2D previously called a helper returning `(x_r, velocity)` and discarded
the second result. The optimized paths retain the **same** state expression:

```python
ratio = (r / t.clamp_min(1e-5)).clamp(0.0, 1.0)
x_r = ratio * x_t + (1.0 - ratio) * predicted_x0
```

No `torch.lerp`, reassociation, special final-step shortcut or epsilon change is
introduced. An additive denominator `t + eps` would require different weights;
it must not be silently replaced by `r / t`.

The six selected weights, EMA conventions, initial noise, RNG consumption,
precision, masks, normalizers, scoring kernels and output ordering are unchanged.
S2G retains its rotation projection **before** interpolation and its valid-length
handling. Training/JVP helpers still calculate their required velocity. Frozen
model and training computations are unchanged.
