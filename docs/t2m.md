# T2M: Full main-model training and fixed-checkpoint inference

This delivery supports **only the Full T2M main model**, with the two fixed
NFE=1 and NFE=4 paper selections below. The portable `run.py` accepts only
`--profile full`; other experiment profiles are rejected. The repository's
current release index covers six main weights across T2M, S2G and M2D.
Historical experiments are not supported delivery entrypoints.

The Full training source is under
`tasks/t2m/snapshots/restart_20260922/code`. Its configuration binding is in
`full_selection_20260923/provenance/manifest.json`. Source hashes are checked
against the bundled manifest.

The released main inference runner uses a separately recorded, inference-only
AST adapter to omit the velocity tensor that the original sampler discarded.
It keeps the exact `ratio*x + (1-ratio)*x0` operation order, time grid, masks,
RNG and model calls. Frozen training/model bytes and the training/JVP velocity
helper remain unchanged. See [sampling](sampling.md) and the per-run
`sampling_optimization` receipt.

## Checkpoints

| Paper view | Fixed selection | Checkpoint SHA256 |
| --- | --- | --- |
| NFE=1 | epoch 7150 | `d1f5848ca560c7812566abef413d1f532e3652f44030af4388ef3447a5932846` |
| NFE=4 | epoch 5850 | `43cecbb482982352754364706ee14932812a22da70606fc1ec412c4663e19c9d` |

## Restore inputs and evaluator paths

Create and activate the T2M environment with [environment setup](environment.md)
before following these paths and commands.

Use Linux with NVIDIA GPUs and the task-specific observed runtime inventory.
`requirements/t2m.reference.txt` is an import reference, not a tested clean
installation recipe; `requirements/t2m.observed.20261010.lock.txt` records package versions.
Do not install or change packages in a live training environment automatically.

Download the current main-model weights from the public `FlashMotion-models`
repository. Obtain authorized data/evaluator assets separately. Use immutable
revisions and SHA/size verification; see [download instructions](published_artifacts.md).
Restore data into exclusive new roots with [dataset restoration](dataset_restore.md).
Do not extract into the Git checkout or an original live training directory.

The current download-only manifests are
`manifests/main6_checkpoints_download.json`,
`manifests/evaluator_assets_download.json` and
`manifests/data_download_authorized_subset.json`. They set `local_path=path`,
so a verified fetch retains the documented HF-relative layout below its new
local stage root.

Create a new complete Flood evaluator root outside Git by safely byte-copying
the verified `third_party/t2m/FloodDiffusion` source, the eight downloaded T2M
evaluator files, and the verified restored raw data. Their required layout is:

```text
<flood_root>/
  metrics/tools/t2m_evaluator.py
  metrics/tools/word_vectorizer.py
  deps/t2m/humanml3d/{text_encoder,movement_encoder,motion_encoder}.pt
  deps/t2m/meta/{mean,std}.npy
  deps/glove/{our_vab_data.npy,our_vab_idx.pkl,our_vab_words.pkl}
  raw_data/HumanML3D/...
  raw_data/BABEL_streamed/...
```

The eight evaluator binaries are stored in the restricted dependency archive at
`t2m/evaluator/FloodDiffusion/deps/...`; they are separate from the two
FlashMotion model checkpoints. Preserve their exact SHA256 from
`manifests/evaluator_assets_download.json`.
Do not download the full upstream 13 GB `deps.zip` when restoring these
already archived minimal assets.

The evaluator manifest has ten files in total: the eight T2M members shown
above, plus S2G's `s2g/evaluator/emage_evaltools/AESKConv_240_100.bin` and
`mean_vel_smplxflame_30.npy`. Those two files belong to S2G's `metric_assets`
root, not the T2M Flood root. They do not include SMPL-X body-model files.
M2D's corrected scorer needs no file from this ten-item evaluator manifest;
its original metric/FK source is preserved in its own task snapshot.

Copy/edit `configs/t2m.paths.example.json` into a private runtime paths JSON.
Relative paths resolve against that JSON's directory:

| Path key / option | Exact required target |
| --- | --- |
| `flood_root` | the new assembled root shown above |
| `data_root` | **exactly** `flood_root/raw_data`; the frozen evaluator enforces this |
| `embedding_root` | restored `data/umt5_xxl_sentence_features` containing the original `embeddings.npy` and `text_index.json` |
| `cohort_root` | restored `cohort311`, with 311 unique `NAME--HASH.npz` files |
| `--canonical` | verified dataset file `t2m/evaluation/canonical_gt.npz` |
| `output_root` | a new external output parent; each run uses a nonexistent child |
| `training_gpus` / `evaluation_gpus` | explicitly owned GPU IDs; eight training ranks or one inference GPU |

Cohort filenames define the frozen BABEL-long validation IDs. The fresh runner
does not reuse their cached method predictions. Never regenerate UMT5 features
and silently substitute a new embedding cache.

The canonical six-field file has SHA256
`9ed7762556bb307bc147358215f849f2f43a20676f703445408269beafd4af9d`.
It contains `ground_truth`, `text`, `segment_key`, `caption`, `episode_id`
and `segment_length`. This hash verifies the distributed input file; the
original launch command and its input bytes were not verified.

## Plan training; use the main runner for paper inference

Default plans use Python >=3.10 standard library only. They do not import
Torch/NumPy, allocate a GPU, start a subprocess, create output directories or
write an inference receipt. Supply and validate all external inputs before execution.

From the release root, with verified local paths filled in:

```bash
python -B tasks/t2m/run.py --profile full --mode train --paths /assets/t2m.paths.json --output /new_outputs/full_train --dry-run
python -B tasks/t2m/reproduce_main.py --paths /assets/t2m.paths.json --nfe 1 --checkpoint /models/full-step-0100100.pt --checkpoint-sha256 d1f5848ca560c7812566abef413d1f532e3652f44030af4388ef3447a5932846 --canonical /data/t2m/evaluation/canonical_gt.npz --output /new_outputs/main1_fresh --gpu 0 --cap .08 --seed 42 --dry-run
python -B tasks/t2m/reproduce_main.py --paths /assets/t2m.paths.json --nfe 4 --checkpoint /models/full-step-0081900.pt --checkpoint-sha256 43cecbb482982352754364706ee14932812a22da70606fc1ec412c4663e19c9d --canonical /data/t2m/evaluation/canonical_gt.npz --output /new_outputs/main4_fresh --gpu 0 --cap .08 --seed 42 --dry-run
python -B -m unittest discover -s tests -p test_t2m_entrypoints.py -v
```

`--execute` launches the native command. Use the task entrypoint and a fresh
output directory for each run; historical controllers and original output
directories belong to the archived experiments.

Use `reproduce_main.py` for the reported paper rows. It keeps generation
matmul precision `high`, re-encoding precision `highest`, embedding batch32,
exact canonical six-field comparison and the original CPU Mixed scorer.
`run.py --mode eval` is only a plan for the original native periodic
evaluator; it is **not** the main-table uniform re-encoding route and its
metrics must not be substituted for the table above. Its supported NFE values
are 1 and 4.

Select an available GPU and coordinate shared use before execution.
The runner does not acquire a shared GPU lock. Its allocator fraction must be
positive and at most .08, and is set before model/evaluator imports. This cap
does not reserve memory or cover all driver/native allocations.

Fresh main output writes `raw/<episode>.npz`, `embeddings.npz`,
`metrics.json`, `reproduction_plan.json` and
`reproduction_receipt.json`. Trusted checkpoint SHA is checked before loading;
step/epoch and actual finite EMA are mandatory. Actual 371-tensor main EMA
digests are NFE1
`67cd6b62ad91423bb3be9ebe4aba6adee2d23fbbfb8460d647922af16f7ef403`
and NFE4
`cfb4b1ccca90fddc4cfa83d0b757b1d28e3ab30510861bda27ba904c16e741ae`,
using the declared sorted name + shape + tensor-byte hash scheme.
Different hash algorithms are not interchangeable.

Every run needs a nonexistent child of `output_root`; cached generation,
overwrite, skip, resume and automatic retry are forbidden. A failed inference
retains partial outputs and `reproduction_failure.json`; preserve it and do
not retry into that directory. Runner completion alone is not paper-number
acceptance: independently compare raw/EMA/source/input bindings and
full-precision metrics.

## Original Full training configuration and random seeds

The Full architecture is BiMamba + AdaIN + condition selection + sum direction
fusion, with `tr_only` objective and full auxiliary losses. Preserve random
initialization, window200, eight ranks, batch288 per rank/global2304,
112000 updates (14 updates per epoch), BF16 and EMA decay .9999.
Loss weights remain FM1, reconstruction .636, velocity .323,
acceleration .2, FK .646 and floor0. The new main-only command scope does not
change model math, optimizer, data order or original training budget.

Training calls Accelerate `set_seed(42, device_specific=True)` before random
decoder construction. The DataLoader generator uses42; NumPy/Python worker
seeds derive from `torch.initial_seed() % 2**32`. Preserve the original
distributed sampler behavior and recorded epoch updates for this task.

Inference keeps the original explicit RNG rather than adding a global reseed:

- Full episode: `stable_seed(42, variant, episode_name, nfe)`.
- Independent single segment:
  `stable_seed(42, variant, episode_name, segment_index, "single", nfe)`.
- Stable seed: SHA256 of `::`-joined payload; first eight bytes interpreted
  little-endian, modulo `2**31`.
- Retrieval: CPU generator `42 + repeat` for20 repeats.
- Diversity: local NumPy `default_rng(42)`.

Sampling noise is passed through its explicit generator. NFE1/NFE4 use
different derived seeds; this is not a shared-noise comparison. Frozen sources
do not force deterministic CUDA algorithms, cuDNN or a CUBLAS workspace policy.
Use the documented precision and environment, and report any numerical
differences. Inference checks do not validate a clean installation or a new
training run on another host.
