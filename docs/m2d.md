# Music-to-Dance · FineDance

Use Linux with NVIDIA GPUs. Complete [environment setup](environment.md),
[artifact downloads](published_artifacts.md) and [dataset restoration](dataset_restore.md)
before running commands from the repository root.

## Checkpoints and data

| Sampling steps | Epoch | Path relative to the model download root |
| --- | --- | --- |
| 1 | 2250 | `m2d/genre/train-2250.pt` |
| 4 | 2100 | `m2d/genre/train-2100.pt` |

Use the [paper checkpoint manifest](https://huggingface.co/yangkaixing/FlashMotion-models/blob/fcf613fa1742b1655c7299e10de8c8dfc332a8c6/checkpoint-sha256.json).

These checkpoints correspond to the FineDance rows in
[paper Table II](https://arxiv.org/html/2611.13626v1):

| Steps | Epoch | FID_k | FID_g | DIV_k | DIV_g | BAS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2250 | 18.32 | 28.48 | 9.48 | 7.54 | 0.237 |
| 4 | 2100 | 15.15 | 21.13 | 9.55 | 6.58 | 0.240 |

The [M2D checkpoint record](../manifests/m2d_paper_checkpoints.json) binds
each row to its file SHA256 and recorded full-precision quality metrics.

Restore the native 35D music features and 151-channel motion data without
changing their embedded normalizers. The decoder uses 147 pose channels after
the 4 contact channels. Evaluation uses 84 ordered, 1020-frame test windows;
training uses 240-frame windows at 30 FPS.

## Training

Copy [`m2d.paths.example.json`](../configs/m2d.paths.example.json) to an external
config directory and replace its paths, hostname and device IDs. Set
`output_root` to a fresh training directory and `generated_code_root` to its
`code/` child. Data, checkpoints and output directories belong outside Git.

```bash
conda activate flashmotion-m2d
python -B tools/verify_code.py
python -B tasks/m2d/run.py train --version genre --paths /path/to/m2d.paths.json --dry-run
python -B tasks/m2d/run.py train --version genre --paths /path/to/m2d.paths.json --prepare
```

`--prepare` creates a new adapted source copy and records every transformation.
The original training entry writes at import time; use the generated controller
rather than running it inside the snapshot directory. Inspect its configured
runtime, resource checks and output paths before launching:

```bash
python /path/to/new-training-root/full_control.py launch genre verify
# After genre/verify/exit.json reports exit_code 0 and both metric JSONs exist:
python /path/to/new-training-root/full_control.py launch genre train
```

Training uses 4 GPUs, batch size 64 per rank (256 total), seed 42 on each rank,
4000 epochs and FP32. The controller retains the native hostname, lock, free-memory
and verification checks. The verification job performs real training updates and
evaluation. The planner's training `--execute` mode is not implemented.

## Test · Inference and Evaluation

Create separate path configs and fresh `evaluation_output_root` directories for
the 1-step and 4-step runs. Each `runtime_evidence` JSON must bind that run's
checkpoint, SHA, inputs, installed runtime and available GPU resources. Use the dedicated-idle resource policy; the exact evidence fields are checked
by [`run.py`](../tasks/m2d/run.py).

For either selected checkpoint, use its corresponding epoch, step count and
SHA from the manifest:

```bash
python -B tasks/m2d/run.py eval --version genre \
  --paths /path/to/m2d.nfe4.paths.json \
  --checkpoint /path/to/models/m2d/genre/train-2100.pt \
  --checkpoint-sha256 83ef84e3455fd12b1fccce5fcce21e8443208ad223eacd2f450d08614b45bcf2 \
  --epoch 2100 --nfe 4 --dry-run
```

Run once with `--prepare` in place of `--dry-run` to create the evaluation copy.
After validating the required current-host evidence, run with `--execute` to
generate and score the 84 clips. The [README](../README.md#-music-to-dance-m2d--finedance)
includes commands for both selected checkpoints.

The evaluator preserves the native model, input order, seed 42, normalizers and
metric kernels. Inference uses [direct endpoint interpolation](sampling.md).
A dry run checks source and configuration only; it does not test GPU execution.

## Source and reproducibility

The [training source](../tasks/m2d/snapshots/genre_20260929/) is verified
against the bundled SHA records.
The adapter prepares a separate copy with configured paths and, for
evaluation, the fixed-checkpoint and sampling adaptations described above.
