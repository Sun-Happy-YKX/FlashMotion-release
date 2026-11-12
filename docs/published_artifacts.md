# Artifact downloads and local paths

## Paper checkpoints

The six paper checkpoints are public in
[`yangkaixing/FlashMotion-models`](https://huggingface.co/yangkaixing/FlashMotion-models/tree/fcf613fa1742b1655c7299e10de8c8dfc332a8c6).
Use immutable revision `fcf613fa1742b1655c7299e10de8c8dfc332a8c6` and
[`main6_checkpoints_download.json`](../manifests/main6_checkpoints_download.json).
M2D uses epoch **2250 / 1-step** and epoch **2100 / 4-step**; their identities
and FineDance paper metrics are in
[`m2d_paper_checkpoints.json`](../manifests/m2d_paper_checkpoints.json).

Run from the code checkout; replace `/safe` with an absolute directory you own
outside the checkout. Create `/safe/receipts` first. Public checkpoint downloads
do not require an HF login.

```bash
python -B tools/fetch_hf.py manifests/main6_checkpoints_download.json --revision fcf613fa1742b1655c7299e10de8c8dfc332a8c6 --stage-root /safe/FlashMotion-models --receipt /safe/receipts/models.json
```

## Data and evaluation dependencies

Datasets and third-party evaluation/body assets require separate access under
their providers' terms. The recorded restricted archives below supply these
dependencies; they are not the public paper-checkpoint download source.

| Archive | Immutable verified revision | Scope |
| --- | --- | --- |
| Restricted dependency archive `yangkaixing/FlashMotion-checkpoints` | `1e1ba4135613fcba36f0abf72de9fcde61ef549a` | ten evaluator dependency files and one neutral SMPL-X dependency; access required |
| HF Dataset `yangkaixing/FlashMotion-data` | `52124dfa623e878aa8e4118d4066eb9135a8ca05` | complete 33 data package/metadata files including exact S2G cache; 35,189,672,706 bytes |

The restricted-archive helper uses the owner's cached HF login. Other users
obtain these dependencies from their original providers and assemble the paths
below. Never put a token in an argument or source file.

```bash
python -B tools/fetch_hf.py manifests/evaluator_assets_download.json --revision 1e1ba4135613fcba36f0abf72de9fcde61ef549a --stage-root /safe/FlashMotion-models --receipt /safe/receipts/evaluators.json
python -B tools/fetch_hf.py manifests/s2g_body_download.json --revision 1e1ba4135613fcba36f0abf72de9fcde61ef549a --stage-root /safe/FlashMotion-models --receipt /safe/receipts/smplx.json
python -B tools/fetch_hf.py manifests/data_download_complete.json --revision 52124dfa623e878aa8e4118d4066eb9135a8ca05 --stage-root /safe/FlashMotion-data-stage --receipt /safe/receipts/data.json
```

Append `--dry-run` for a no-network/no-write plan. A live fetch verifies every
SHA/size, refuses mismatching existing files and writes a new receipt. The
download manifests use **`local_path = path`**, so downloaded relative paths
match HF paths.

## Task roots after download

All assembly is byte-preserving copy into fresh external directories, with
source/destination SHA/size checked and no overwrite or links. Do not merge
over a live project. Follow [dataset restoration](dataset_restore.md) for tar
validation, exclusive package roots and copy-only merging; that tool does not
claim to verify a later manual merge. Unverified Torch/pickle loading is unsafe.

| Task | Required mapping |
| --- | --- |
| T2M | Build a new complete `flood_root` from frozen `third_party/t2m/FloodDiffusion` evaluator code, `<models>/t2m/evaluator/FloodDiffusion/deps/`, and verified BABEL/HumanML3D `raw_data/`; `data_root` must equal `flood_root/raw_data`. `embedding_root` is restored `data/umt5_xxl_sentence_features`; `cohort_root` is restored `cohort311/`; canonical GT is `<data-stage>/t2m/evaluation/canonical_gt.npz`. |
| S2G | `raw_root` is restored `raw/beat_english_v2.0.0`; the verified exact-cache package restores `cache/`. Copy the two `<models>/s2g/evaluator/emage_evaltools/` files and the permitted `s2g/body_models/smplx_models/` tree into a fresh `metric_assets` root; evaluator source is `third_party/s2g/emage_evaltools`. |
| M2D | `data_root` is the freshly merged canonical/FineDance task root with `data/dataset_backups_stable_v2/{train,test}_tensor_dataset.pkl`. Copy verified `data_source/test` into a new `data/test`, including `motions_sliced` and `baseline_feats`; no symlinks. There is no M2D body-model/evaluator binary in the ten-file dependency set. |

Use the checkpoint paths and hashes in the six-item manifest, with each
task's documented EMA convention, evaluation protocol and seed rules.

S2G's accepted `infer.py` intentionally pins the original config bytes and its
absolute roots. The table above maps assets for a separately reviewed relocated
deployment/planner; it does **not** make the original bounded replay accept new
paths or prove new-host inference. Original-host inference was checked, clean
installation/relocation was not. Keep that distinction when reproducing.

## Additional audio/body dependencies

HuBERT and vocabulary selective official download instructions, revisions and
SHA256 are recorded in `docs/runtime/evaluator_assets.20261010.json`. Use the
recorded provider and revisions for reproduction.

The permission record covers owner-confirmed private distribution of the
neutral SMPL-X file and cache `pose_mean`. It does not authorize public
redistribution, and the supporting grant document has not been independently
reviewed. The [official model license](https://smpl-x.is.tue.mpg.de/modellicense.html) applies.
The archive uses the exact 167,264,530-byte original-evaluation file, SHA256
`bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74`,
not an unverified HF mirror or a newer substitute. Users can alternatively get
their own file from the [official download page](https://smpl-x.is.tue.mpg.de/download.php).
