# Linux conda setup for the six main checkpoints

Create one environment per task from the repository root. These recipes target
Linux x86_64, Python 3.10.12, PyTorch 2.7.1 with CUDA 12.8 and the observed native
source revisions. They add an installation route; a fresh conda solve, extension
build and new-host six-checkpoint replay have **not been tested**. Successful
package installation alone does not reproduce the reported metrics.

The `*.observed.20261010.lock.txt` files record package versions. The smaller `*.runtime.txt` files
pin the main task import closures; they are not complete transitive/build locks.
The YAML files provide bootstrap tools, CUDA SDK and, where needed, audio tools.
GCC 12 is a compatible build-tool choice, not a recorded original compiler pin.
Conda bootstrap package builds and the NVIDIA driver remain host dependencies.

## Create and activate

Choose the block for the task you will run. The CUDA toolkit does not install a
host NVIDIA driver. Use a CUDA-capable GPU and a driver compatible with CUDA 12.8;
the recorded reference runtime uses H200 GPUs and driver 580.95.05.
Validate compatibility on the hardware you will use.
The commands assume conda is already installed on that Linux host and Bash is
initialized for activation; if needed, run `conda init bash` and restart the
shell as described in [conda's initialization guide](https://docs.conda.io/projects/conda/en/stable/commands/init.html).

```bash
conda env create -f environments/t2m.yaml
conda activate flashmotion-t2m
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=numpy,scipy,scikit-learn,numba,llvmlite -r requirements/t2m.runtime.txt
```

```bash
conda env create -f environments/s2g.yaml
conda activate flashmotion-s2g
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=numpy,scipy,scikit-learn,numba,llvmlite -r requirements/s2g.runtime.txt
```

```bash
conda env create -f environments/m2d.yaml
conda activate flashmotion-m2d
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=numpy,scipy,scikit-learn,numba,llvmlite -r requirements/m2d.runtime.txt
```

The torch wheel index is the [official PyTorch 2.7.1 CUDA 12.8 route](https://pytorch.org/get-started/previous-versions/).
The SDK channel fixes NVIDIA's CUDA 12.8.0 release label, whose observed compiler
is `nvcc V12.8.61`; see [NVIDIA's conda installation instructions](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#conda-installation).
NumPy 1.23.5 satisfies the runtime bounds of [SciPy 1.15.3](https://github.com/scipy/scipy/blob/v1.15.3/pyproject.toml)
and [scikit-learn 1.7.2](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/pyproject.toml).
Their isolated source-build NumPy requirements are different, hence the wheel
selection above. Do not upgrade NumPy independently of the pinned audio stack.

Primary PyPI metadata confirms CPython 3.10 Linux wheels for those three numeric
pins, Numba 0.56.4, llvmlite 0.39.1, [PyYAML 6.0.3](https://pypi.org/project/PyYAML/6.0.3/)
and tokenizers 0.22.1. This is package-metadata inspection, not a full resolver
or clean-install test. The observed [transformers 4.57.0](https://pypi.org/project/transformers/4.57.0/)
release is yanked for an upstream setup/installation issue. Its exact wheel is
still published; an exact pin may select it with a warning. We retain the
observed version; clean installation of this pin has not been verified. TextGrid, PyOpenGL, p-tqdm and iopath
may require their small source distributions; the wheel-only flag above applies
only to the named numeric packages.

## Build native providers after torch

Run this in each activated task environment. The source commits below are the
editable source HEADs captured in all three runtime inventories. Their package
versions are Mamba 2.2.6.post3 and causal-conv1d 1.5.3.post1. Force source builds
against the already installed torch and toolkit; no native binary from the
original machine is redistributed or asserted bit-identical.

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
nvcc --version
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@857842b10546d86aaf05c52570a51fba2eec04d2"
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@10b5d6358f27966f6a40e4bf0baa17a460688128"
python -B tools/patch_mamba_interface.py --task t2m
python -B tools/patch_mamba_interface.py --task t2m --apply
```

The build switches and torch-first requirement come from the pinned
[causal-conv1d setup](https://github.com/Dao-AILab/causal-conv1d/blob/857842b10546d86aaf05c52570a51fba2eec04d2/setup.py)
and [Mamba setup](https://github.com/state-spaces/mamba/blob/10b5d6358f27966f6a40e4bf0baa17a460688128/setup.py).
Substitute `s2g` or `m2d` in the two patch commands for their environments. Run
them only in the newly created environment, before importing Mamba, never in an
existing environment used by a job.

The original runtimes used a locally modified Mamba Python interface, not just
the upstream Git HEAD. The [exact recorded file and minimal diff](../third_party/native_patches/README.md)
restore its one out-of-place `addmm` repair. The stdlib-only patch command first
inspects without writing; `--apply` requires the expected task conda prefix,
distribution version, pip Git commit and exact upstream file hash, retains an
upstream-byte backup, and verifies the recorded output hash. Unknown files fail
closed. It changes no CUDA binary and imports no native provider. Both causal
interface hashes match their pinned upstream revision; this comparison does not
prove all other files in the original editable worktrees were clean.

M2D additionally imports PyTorch3D transforms throughout its native training and
scoring closure. Build its observed 0.7.8 source revision in `flashmotion-m2d`:

```bash
FORCE_CUDA=1 MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
python -m pip check
```

PyTorch3D's [pinned source instructions](https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/INSTALL.md)
document `FORCE_CUDA` and building against installed torch. That document's
tested torch list predates 2.7.1; this release observed the combination working
on the original host; a clean build has not been tested. T2M's main
`t2m_long` route and S2G's full55 route do not import PyTorch3D.

For a headless build targeting the recorded H200 hardware, set
`export TORCH_CUDA_ARCH_LIST="9.0"` before the PyTorch3D command. On another GPU,
select its actual [NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus)
instead; never copy `9.0` for unrelated hardware. PyTorch's
[extension instructions](https://docs.pytorch.org/docs/2.7/cpp_extension.html)
describe this build-target selection. The pinned Mamba/causal builders supply
their own architecture flags, including compute 90 and CUDA-12.8-gated 100/120;
the presence of those flags is not validation on Blackwell or any new host.

## Inspect the activated environment

```bash
python -B tools/check_environment.py --task t2m
# Substitute s2g or m2d for their environments.
python -m pip check
```

The default checker reads distribution metadata and runs only `nvcc --version`,
`ffmpeg -version` and `ffprobe -version` when present. It does not import torch,
native providers or task code, query GPUs, download/install anything or write
proofs. `MATCH` means the listed metadata/tool checks match, not ML acceptance.
Native pip Git commit metadata is reported without URLs/authentication fields;
a known revision mismatch fails the check. Editable local installs may lack a
commit in pip metadata, which remains explicitly unknown rather than a verified
source revision. The original inventories separately captured their Git HEADs.

On a GPU selected for your own validation, use `--native` to import/hash actual
providers, or `--cuda` to additionally allocate and synchronize one tiny CUDA
tensor. Provider imports may initialize a driver. These checks do not launch a
model, test MeanFlow JVPs or check datasets. Native checks compare the three
recorded interface source hashes for every task. S2G additionally hard-gates
these hashes in its bundled higher-order JVP adapter; its GPU/JVP and
numerical proofs remain training inputs.
S2G precheck rendering additionally needs a working host EGL/OpenGL setup.

## Restore artifacts and run the selected task

Follow [verified downloads and task root mapping](published_artifacts.md) and
[dataset restoration](dataset_restore.md), placing data/weights outside Git.
The runtime packages include `huggingface-hub` for `tools/fetch_hf.py`; private
access uses your normal cached HF login. Copy the corresponding `configs/*.paths.example.json`
to an external path and fill in absolute data, model, runtime and fresh output
paths. Installing Python packages does not provide separately licensed assets.

| Environment | Cached inputs and selected models | Training / main-checkpoint inference |
| --- | --- | --- |
| `flashmotion-t2m` | BABEL/HumanML3D roots, precomputed UMT5 embeddings, cohort311, canonical GT and Flood evaluator dependencies; epochs 7150 / 5850 | [T2M guide](t2m.md): native trainer planner and `tasks/t2m/reproduce_main.py` |
| `flashmotion-s2g` | Exact BEAT2 cache, full55 statistics/manifest, SMPL-X and AESK dependencies; epochs 3200 / 1500 | [S2G guide](s2g.md): native train/preprocess planner and `tasks/s2g/infer.py` |
| `flashmotion-m2d` | Canonical/FineDance cache and 84 test motion/features; epochs 2250 / 2100 | [M2D guide](m2d.md): prepared native training copy and repaired checkpoint scorer |

These are the six fixed 1-step/4-step main models. Preserve each task's native
seed, EMA, normalizer, sample order and precision contracts. S2G's current replay
binds its original config, absolute roots and numerical proof; a newly installed
conda environment alone does not clear new-host relocation. M2D execution also
requires a current host/input/native-runtime proof. Use the task guides' planning
and preparation routes before executing native commands in fresh output roots.
