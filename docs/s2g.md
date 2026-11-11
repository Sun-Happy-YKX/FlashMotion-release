# Speech-to-Gesture · BEAT2

Use Linux with NVIDIA GPUs. Complete [environment setup](environment.md),
[artifact downloads](published_artifacts.md) and [dataset restoration](dataset_restore.md).
Commands below run from the repository root.

## Checkpoints and data

| Sampling steps | Epoch | Path relative to the model download root |
| --- | --- | --- |
| 1 | 3200 | `s2g/full55_r2/epoch_03200.pt` |
| 4 | 1500 | `s2g/full55_r2/epoch_01500.pt` |

Use [`main6_checkpoints_download.json`](../manifests/main6_checkpoints_download.json)
for SHA256 and the immutable model revision.

The native pipeline uses the official English split: 1093 training, 106
validation and 265 test recordings at 30 FPS. Inputs combine 1024D HuBERT speech
features and 300D text features. Motion has 333 channels: 55 local-joint 6D
rotations and 3 root-motion channels. Keep the original representation,
normalizer and full test-clip budgets.

Restore the complete exact cache from
[`data_download_complete.json`](../manifests/data_download_complete.json).
Evaluation also requires the EMAGE evaluator source, its 2 metric assets and
the neutral SMPL-X model; their paths and access terms are in the
[artifact guide](published_artifacts.md).

## Training and preprocessing

Copy [`s2g.paths.example.json`](../configs/s2g.paths.example.json) to an external
config directory and replace its placeholders. The planner verifies source
hashes and prints native commands; it does not launch jobs.

```bash
conda activate flashmotion-s2g
python -B tools/verify_code.py
python -B tasks/s2g/run.py preprocess --paths /path/to/s2g.paths.json --dry-run
python -B tasks/s2g/run.py train --paths /path/to/s2g.paths.json \
  --mode full55 --batch-per-gpu 32 --dry-run
```

Preprocessing starts with shard 0 to create the manifest. Complete all feature
shards before the train-only statistics pass. To create a new native config
draft in an existing external directory:

```bash
python -B tasks/s2g/run.py train --paths /path/to/s2g.paths.json \
  --write-config --native-config /path/to/new/s2g.training.json
```

The draft has `template_only_not_launchable=true`. The native trainer requires
host/runtime, numerical-environment, higher-order JVP, precheck and batch proofs
bound to the deployed source and config. Path substitution alone is insufficient;
the native [`verify.py`](../tasks/s2g/snapshots/full55_r2/code/verify.py) validates these inputs.
Once the deployment is validated, the native launch command is:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/s2g/snapshots/full55_r2/code/train.py \
  --config /path/to/validated/s2g.training.json --mode full55 --batch-per-gpu 32
```

Training uses 8 GPUs, batch size 32 per rank (256 total), 240-frame windows and
stride 120, 10000 configured epochs and a checkpoint every 50 epochs. Native
initialization uses seed 42; training then uses `42 + rank`.

## Test · Inference and Evaluation

[`infer.py`](../tasks/s2g/infer.py) generates all 265 predictions and scores
them with the native full55 evaluator. The [README](../README.md#-speech-to-gesture-s2g--beat2)
provides separate 1-step and 4-step commands and their shared input arguments.
Use a fresh output directory, one explicitly selected GPU and the original
per-clip seed formula `42 + int(sha256(id).hexdigest()[:8], 16)`.

The accepted replay binds the original config bytes, absolute input roots,
checkpoint, cache, normalizer and source hashes. **Deployment at new paths has
not been validated.** Supplying a planner-generated training config does not
make it a valid inference config. Supply SHA256-bound source, config, checkpoint,
normalizer, manifest and cache-integrity files through the arguments in the README.
Use `OPENBLAS_CORETYPE=Haswell`, single-thread BLAS/OpenMP settings and a positive
GPU allocator fraction no greater than 0.08. The source proof is
`manifests/frozen_source_collection.json`; compute its SHA256 from the deployed file.
Cache integrity must match the original precheck binding.

`--dry-run` verifies inputs and prints the plan without importing model code or
writing outputs. `--execute` loads the selected checkpoint, generates full
clips and evaluates them. Inference preserves the native
[direct-interpolation sampler](sampling.md), without a GT seed pose or clip
truncation.

## Source and reproducibility

The [full55-r2 source](../tasks/s2g/snapshots/full55_r2/code/) and third-party
evaluator files are verified against the bundled SHA records.
