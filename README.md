<p align="center">
  <img src="assets/flashmotion-wordmark.png" alt="FlashMotion" width="560">
</p>
<h3 align="center">Few-step 3D motion generation across text, speech and music</h3>

<p align="center">
  Kaixing Yang, Xulong Tang<sup>‡</sup>, Ronghui Li<sup>‡</sup>, Xiangyue Zhang, Ziqiao Peng, Mingshuang Luo, Chenxi Zhao, Haoyu Wu,
  Ju Fan, Hongyan Liu<sup>†</sup>, Puwei Wang<sup>†</sup>, Zijiao Zeng<sup>†</sup>, and Jun He
</p>
<p align="center"><sup>†</sup> Corresponding authors. &nbsp;·&nbsp; <sup>‡</sup> Project Leader</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.13626"><img src="https://img.shields.io/badge/Paper-arXiv%3A2611.13626-b31b1b?logo=arxiv" alt="Paper"></a>
  <a href="https://sun-happy-ykx.github.io/FlashMotion-Project/"><img src="https://img.shields.io/badge/Project_Page-Website-2463a5?logo=githubpages" alt="Project_Page"></a>
  <a href="https://github.com/Sun-Happy-YKX/FlashMotion-release"><img src="https://img.shields.io/badge/Code-GitHub-24292f?logo=github" alt="Code"></a>
  <a href="https://huggingface.co/yangkaixing/FlashMotion-models"><img src="https://img.shields.io/badge/Checkpoints-Hugging_Face-ffd21e?logo=huggingface" alt="Checkpoints"></a>
  <a href="https://huggingface.co/datasets/yangkaixing/FlashMotion-data"><img src="assets/flashmotion-data-badge.svg" alt="Data on Hugging Face (access required)"></a>
</p>

FlashMotion supports three tasks with **six pretrained checkpoints**: one
1-step and one 4-step checkpoint per task. Training, inference and evaluation code
are included in this public code release. Dataset archives and third-party
evaluation assets are subject to their original access terms.

## 🎯 Three tasks

| Task | Dataset | Train / inference guide |
| --- | --- | --- |
| 💬 Text-to-motion (T2M) | BABEL-long | [T2M](docs/t2m.md) |
| 🗣️ Speech-to-gesture (S2G) | BEAT2 | [S2G](docs/s2g.md) |
| 🎵 Music-to-dance (M2D) | FineDance | [M2D](docs/m2d.md) |

Each task guide includes training, inference and checkpoint details.

## 📦 Code, data and checkpoints

- **Code:** task-specific training and inference source, configuration templates,
  evaluator code and documentation. No large data or model binaries in Git.
- **Data:** [FlashMotion-data on Hugging Face](https://huggingface.co/datasets/yangkaixing/FlashMotion-data) (access required).
  The archive contains 33 verified files (**35.19 GB**) across `t2m/`, `s2g/` and
  `m2d/`, including the preprocessed BEAT2 cache for speech-to-gesture.
- **Checkpoints:** the [six paper checkpoints](https://huggingface.co/yangkaixing/FlashMotion-models/tree/fcf613fa1742b1655c7299e10de8c8dfc332a8c6).
  Evaluator assets and the SMPL-X body model come from their upstream providers.

Use the [download instructions](docs/published_artifacts.md)
for immutable revisions, manifests and local directory mapping, then follow
[dataset restoration](docs/dataset_restore.md).

Dataset access: [BABEL](https://babel.is.tue.mpg.de/), [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2),
[FineDance](https://github.com/li-ronghui/FineDance), and [SMPL-X](https://smpl-x.is.tue.mpg.de/).

## 🚀 Setup and usage

Use **Linux with NVIDIA GPUs** for native model execution. Each task has its own
environment and input contract; start with the corresponding guide above.

1. Obtain the datasets and required third-party assets from their providers. The
   [artifact guide](docs/published_artifacts.md) documents the original archived layout.
2. Download and verify the recorded artifacts into directories outside Git.
3. Follow the task's environment instructions and fill in its local paths
   template from `configs/`.
4. Inspect the generated command before executing it; use fresh output
   directories and GPU resources that you own.

Start with the [environment setup](docs/environment.md): task-specific Conda
YAMLs, pinned dependency TXT files, activation commands and environment checks.
Keep credentials in the standard Git/HF credential store, never in source files.

## 🏋️ Training · 🎬 Inference · 📏 Evaluation

Choose a task below. Each **Test** command generates motion and computes the
task's evaluation metrics. Both **1-step** and **4-step** commands name their corresponding weights
from the [paper checkpoint manifest](https://huggingface.co/yangkaixing/FlashMotion-models/blob/fcf613fa1742b1655c7299e10de8c8dfc332a8c6/checkpoint-sha256.json).

First complete [environment setup](docs/environment.md),
[artifact downloads](docs/published_artifacts.md) and
[dataset restoration](docs/dataset_restore.md). Copy and fill in the matching
[path template](configs/), and set these paths from the repository root:

```bash
export DATA_ROOT=/path/to/restored/FlashMotion-data
export MODEL_ROOT=/path/to/downloaded/FlashMotion-models
export CONFIG_ROOT=/path/to/your/local/configs
export OUTPUT_ROOT=/path/to/new/FlashMotion-runs
mkdir -p "$OUTPUT_ROOT"
```

Use a fresh output directory for each run. The **base seed is 42**, with each
task's native rank, worker and per-sample rules preserved. The main samplers use
[direct endpoint interpolation](docs/sampling.md).

### 💬 Text-to-Motion (T2M) · BABEL

[Full T2M guide](docs/t2m.md) · Training uses 8 GPUs; testing uses 1 GPU.
Fill in `t2m.paths.json` with the motion data, UMT5 feature cache, FloodDiffusion
evaluator and cohort311 paths; set its `output_root` to `$OUTPUT_ROOT`.

**Training**

```bash
conda activate flashmotion-t2m
python tasks/t2m/run.py --profile full --mode train \
  --paths "$CONFIG_ROOT/t2m.paths.json" \
  --output "$OUTPUT_ROOT/t2m_train" --execute
```

**Test (Inference + Evaluation)**

```bash
# 1-step · epoch 7150
python tasks/t2m/reproduce_main.py \
  --paths "$CONFIG_ROOT/t2m.paths.json" --nfe 1 \
  --checkpoint "$MODEL_ROOT/t2m/t2m_adain_selection_8000ep_20260923/adain_selection/checkpoints/step-0100100.pt" \
  --checkpoint-sha256 d1f5848ca560c7812566abef413d1f532e3652f44030af4388ef3447a5932846 \
  --canonical "$DATA_ROOT/t2m/evaluation/canonical_gt.npz" \
  --output "$OUTPUT_ROOT/t2m_test1" --gpu 0 --seed 42 --execute
```

```bash
# 4-step · epoch 5850
python tasks/t2m/reproduce_main.py \
  --paths "$CONFIG_ROOT/t2m.paths.json" --nfe 4 \
  --checkpoint "$MODEL_ROOT/t2m/t2m_adain_selection_8000ep_20260923/adain_selection/checkpoints/step-0081900.pt" \
  --checkpoint-sha256 43cecbb482982352754364706ee14932812a22da70606fc1ec412c4663e19c9d \
  --canonical "$DATA_ROOT/t2m/evaluation/canonical_gt.npz" \
  --output "$OUTPUT_ROOT/t2m_test4" --gpu 0 --seed 42 --execute
```

Use `reproduce_main.py` for the main-table protocol; the training planner's
native evaluation mode is a different route.

### 🗣️ Speech-to-Gesture (S2G) · BEAT2

[Full S2G guide](docs/s2g.md) · Training uses 8 GPUs; testing uses 1 GPU.
Prepare the native training config and runtime records using the guide before
launching. The native trainer checks the configured host, assets and output
identity. This release preserves the original config/path bindings;
new-host relocation has not yet been validated.

**Training**

```bash
conda activate flashmotion-s2g
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/s2g/snapshots/full55_r2/code/train.py \
  --config "$CONFIG_ROOT/s2g.training.json" --mode full55 --batch-per-gpu 32
```

`tasks/s2g/run.py train` previews a training plan; it does not launch training.

**Test (Inference + Evaluation)**

Set `S2G_SOURCE`, `S2G_CACHE`, `S2G_RAW`, `S2G_METRIC_ASSETS` and
`S2G_EVALUATOR_SOURCE` to the deployed locations. Set `SOURCE_PROOF`,
`SOURCE_PROOF_SHA`, `CACHE_INTEGRITY` and `CACHE_INTEGRITY_SHA` to the verified
records described in the [inference guide](docs/s2g.md).

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Shared inputs; set once for both checkpoints.
S2G_TEST_ARGS=(
  --execute --seed 42
  --source-dir "$S2G_SOURCE"
  --source-proof "$SOURCE_PROOF" --source-proof-sha256 "$SOURCE_PROOF_SHA"
  --config "$S2G_SOURCE/config.target.json"
  --config-sha256 512f7c15118fb8afa66227038aba38c129a045d321e4a30eefa24dc7366af4f1
  --normalizer "$S2G_CACHE/stats_full55.npz"
  --normalizer-sha256 cc65a7737f6ef09a727987ca73cde8cb320c66b16645d5bc0579ffd57fbe2e28
  --manifest "$S2G_CACHE/manifest.json"
  --manifest-sha256 7f2e1ed9df0d5671cfaff08f69f96cc24eeb0aec9400f1515b81f33161dfbf4c
  --cache-integrity "$CACHE_INTEGRITY" --cache-integrity-sha256 "$CACHE_INTEGRITY_SHA"
  --cache-root "$S2G_CACHE" --raw-root "$S2G_RAW"
  --metric-assets "$S2G_METRIC_ASSETS" --evaluator-assets "$S2G_EVALUATOR_SOURCE"
)
```

```bash
# 1-step · epoch 3200
python tasks/s2g/infer.py "${S2G_TEST_ARGS[@]}" \
  --checkpoint "$MODEL_ROOT/s2g/full55_r2/epoch_03200.pt" \
  --checkpoint-sha256 a6f5dcfab53dace000485ef475b1e714c21c5441209ef3343326d43befda2c2b \
  --expected-epoch 3200 --steps 1 --output "$OUTPUT_ROOT/s2g_test1"

# 4-step · epoch 1500
python tasks/s2g/infer.py "${S2G_TEST_ARGS[@]}" \
  --checkpoint "$MODEL_ROOT/s2g/full55_r2/epoch_01500.pt" \
  --checkpoint-sha256 6834c61dc355ebb740d58edfb7649c9cf26dcb6401c72b4be25423f5b278c002 \
  --expected-epoch 1500 --steps 4 \
  --output "$OUTPUT_ROOT/s2g_test4"
```

### 🎵 Music-to-Dance (M2D) · FineDance

[Full M2D guide](docs/m2d.md) · Training uses 4 GPUs; testing uses 1 GPU.
Fill in `m2d.paths.json`; its training `output_root` must be
`$OUTPUT_ROOT/m2d_train`.
Prepare and verify the runtime as described in the guide.

**Training**

```bash
conda activate flashmotion-m2d
python tasks/m2d/run.py train --version genre \
  --paths "$CONFIG_ROOT/m2d.paths.json" --prepare
python "$OUTPUT_ROOT/m2d_train/full_control.py" launch genre verify
```

The verification job runs asynchronously. Wait for
`$OUTPUT_ROOT/m2d_train/genre/verify/exit.json` to report `exit_code: 0`
and for both verification metric JSONs before launching training:

```bash
python "$OUTPUT_ROOT/m2d_train/full_control.py" launch genre train
```

Use the generated controller, which retains the native verify and launch checks.
The planner's training `--execute` mode is not implemented.

**Test (Inference + Evaluation)**

Create `m2d.nfe1.paths.json` and `m2d.nfe4.paths.json` from the path template.
Give them separate fresh `evaluation_output_root` directories, for example
`$OUTPUT_ROOT/m2d_test1` and `$OUTPUT_ROOT/m2d_test4`. Each config's
`runtime_evidence` must bind its own checkpoint and SHA.

```bash
M2D_1STEP_ARGS=(
  --version genre --paths "$CONFIG_ROOT/m2d.nfe1.paths.json"
  --checkpoint "$MODEL_ROOT/m2d/genre/train-2250.pt"
  --checkpoint-sha256 b6f3860cc9c3a5ce8d62bbe419fa50f145654a30af5cffdedc94461285f483e7
  --epoch 2250 --nfe 1
)
M2D_4STEP_ARGS=(
  --version genre --paths "$CONFIG_ROOT/m2d.nfe4.paths.json"
  --checkpoint "$MODEL_ROOT/m2d/genre/train-2100.pt"
  --checkpoint-sha256 83ef84e3455fd12b1fccce5fcce21e8443208ad223eacd2f450d08614b45bcf2
  --epoch 2100 --nfe 4
)
```

Prepare each evaluation run once:

```bash
python tasks/m2d/run.py eval "${M2D_1STEP_ARGS[@]}" --prepare
python tasks/m2d/run.py eval "${M2D_4STEP_ARGS[@]}" --prepare
```

After the guide's current-host `runtime_evidence` is ready for each run,
generate and score the 84 test clips:

```bash
python tasks/m2d/run.py eval "${M2D_1STEP_ARGS[@]}" --execute  # 1-step · epoch 2250
python tasks/m2d/run.py eval "${M2D_4STEP_ARGS[@]}" --execute  # 4-step · epoch 2100
```

## 🧩 Directory Structure

The three tasks live under `tasks/`. Start with each task's guide for training
and paper-checkpoint inference; download data and weights outside this checkout.

```text
FlashMotion/
├── tasks/
│   ├── t2m/                      # Text-to-motion · BABEL
│   │   ├── run.py                # Training / native-evaluation planner
│   │   ├── reproduce_main.py     # Selected 1-step / 4-step paper inference
│   │   └── snapshots/            # Original training and model source
│   ├── s2g/                      # Speech-to-gesture · BEAT2
│   │   ├── run.py                # Preprocess / training / evaluation planner
│   │   ├── infer.py              # Selected 1-step / 4-step paper inference
│   │   └── snapshots/            # Original full55 training and model source
│   └── m2d/                      # Music-to-dance · FineDance
│       ├── run.py                # Training planner / native evaluation
│       ├── adapter.py            # Path and source-copy adapter
│       └── snapshots/            # Original genre training and model source
├── configs/                      # Local path templates for each task
├── environments/                 # Task-specific Conda environment definitions
├── requirements/                 # Task-specific environment records
├── docs/                         # Training, inference and download guides
├── manifests/                    # Selected weights, SHA256 and checks
├── third_party/                  # Evaluator source and upstream notices
├── tools/                        # Download, restore and verification helpers
├── tests/                        # CPU-only helper tests
└── README.md
```

## 🙏 Acknowledgments

We thank [FlowerDance](https://github.com/XulongT/FlowerDance),
[SemTalk](https://github.com/Xiangyue-Zhang/SemTalk),
[FloodDiffusion](https://github.com/AlayaLab/FloodDiffusion),
[text-to-motion](https://github.com/EricGuo5513/text-to-motion) and
[EMAGE evaltools](https://huggingface.co/H-Liu1997/emage_evaltools) for their
open-source implementations and evaluation resources. Additional attribution
is preserved in [third-party notices](third_party/NOTICE.md).

## 📚 Citation

If you find FlashMotion useful in your research, please cite:

```bibtex
@misc{yang2026flashmotion,
  title={FlashMotion: Ultra-Fast Long-Term 3D Motion Generation},
  author={Yang, Kaixing and Tang, Xulong and Li, Ronghui and Zhang, Xiangyue and Peng, Ziqiao and Luo, Mingshuang and Zhao, Chenxi and Wu, Haoyu and Fan, Ju and Liu, Hongyan and Wang, Puwei and Zeng, Zijiao and He, Jun},
  year={2026},
  eprint={2611.13626},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.13626}
}
```

## 📄 License and Usage

The original FlashMotion code is released under the [MIT License](LICENSE).
Third-party code, datasets, evaluation assets, and body models retain their
respective licenses and access requirements. See the
[third-party notices](third_party/NOTICE.md) and
[artifact guide](docs/published_artifacts.md) for details.
