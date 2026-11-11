# Third-party evaluator attribution

This notice is generated for this release; it is not represented as an upstream NOTICE.
See `manifests/third_party_source_collection.20261010.json` for original-host paths,
pinned upstream revisions, exact byte hashes, and source/license evidence.

## T2M

The three files in `t2m/FloodDiffusion/metrics/tools` are byte-identical to
AlayaLab/FloodDiffusion commit `b611d6c4e24357a90066034935ed445179723b37`
and the actual DMD01 evaluator. Its preserved `THIRD_PARTY_LICENSES.md`
identifies these adapted files as EricGuo5513/text-to-motion, MIT.
The MIT text is retained in `t2m/LICENSE.text-to-motion.MIT.txt`
(Copyright (c) 2022 Chuan Guo).
FloodDiffusion's Apache LICENSE applies to its original code, not as a
replacement for that MIT attribution.

All three native evaluator encoder classes are implemented in
`t2m_evaluator.py`; no guessed model files have been added. The native
FlashMotion adapter imports these three tools directly, not Flood's
Lightning/torchmetrics `metrics/t2m.py` wrapper.

## S2G

The four actual vendor files in `s2g/emage_evaltools` are byte-identical to
the owner release H-Liu1997/emage_evaltools revision
`a57d1dfb1162c2a9cc20013f0ab212c21f211e78`.
That pinned repository's model card declares `apache-2.0`; the original
card bytes/hash are retained as evidence in the collection receipt.
The standard Apache 2.0 text from apache.org is retained as
`s2g/LICENSE.Apache-2.0.txt`.

Additional inherited terms are retained without being overridden:

- `decoders.py` identifies EricGuo5513/TM2T (MIT); its header and
  `s2g/LICENSE.TM2T.MIT.txt` retain the original terms.
- `skeleton_DME.py` retains the full 2020 DeepMotionEditing copyright,
  two redistribution conditions, and disclaimer in its original source.
  The actual text is BSD-2-Clause-style, not a three-condition BSD license.

The original vendor directory contains four source files, not eight, and
has no standalone LICENSE/NOTICE. We did not substitute licenses from
a different old StreamTalk implementation.

## Asset and publishing boundary

Evaluator weights, HuBERT, vocabulary, numerical cache outputs, datasets,
and SMPL-X models are not all covered by the code repository's own license.
Exact asset hashes and selective official download instructions are in
`docs/runtime/evaluator_assets.20261010.json`. The owner later confirmed
permission for the private SMPL-X and exact S2G cache delivery, including its
pose mean; see `manifests/s2g_private_dependency_permission.20261010.json` and
`docs/published_artifacts.md` for the completed private uploads. This confirmation
does not override upstream terms or authorize making these assets public.

## Native provider repair

The exact recorded Mamba Python interface, minimal higher-order-AD repair and
upstream Apache-2.0 license are retained under `native_patches/`. Its
[source notice](native_patches/README.md) identifies the upstream commit,
original/modified hashes and explicit installation boundary. Original frozen
task source and the twelve evaluator/license bytes above remain unchanged.
