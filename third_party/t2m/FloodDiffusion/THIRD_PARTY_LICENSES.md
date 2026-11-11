# Third-Party Code Attribution

This project uses code from the following open-source projects. All copyright notices are preserved in the source files.

---

## 1. Quaternion Operations

**File:** `utils/math/quaternion.py`  
**Source:** Facebook Research - VideoPose3D  
**Repository:** https://github.com/facebookresearch/VideoPose3D  
**License:** BSD-3-Clause  
**Modifications:** Adaption for numpy

Copyright notice from file:

```
Copyright (c) 2018-present, Facebook, Inc. All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
```

---

## 2. Alibaba WAN Model Components

**Files:**

-   `models/tools/t5.py`
-   `models/tools/wan_model.py`
-   `models/tools/wan_model_cross_rope.py`
-   `models/tools/wan_vae.py`
-   `models/tools/wan_vae_1d.py`
-   `models/tools/attention.py`
-   `models/tools/tokenizers.py`

**Source:** Alibaba Wan Team  
**Repository:** https://github.com/Wan-Video/Wan2.2
**License:** Apache 2.0  
**Modifications:** Modified for streaming motion generation (causal attention, stream mode, context length handling)

Copyright notice from files:

```
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
```

**Note:** `t5.py` is also based on Hugging Face Transformers (Apache 2.0):

```
Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
Licensed under the Apache License, Version 2.0
```

---

## 3. Text-to-Motion Evaluation Metrics

**Files:**

-   `metrics/tools/t2m_evaluator.py`
-   `metrics/tools/utils.py`
-   `metrics/tools/word_vectorizer.py`

**Source:** text-to-motion  
**Repository:** https://github.com/EricGuo5513/text-to-motion  
**License:** MIT  
**Modifications:** Adapted for this project's evaluation pipeline

These files implement standard text-to-motion metrics: R-precision, FID, diversity, and multimodality.

---

## License Compatibility

All third-party licenses are compatible with this project's Apache 2.0 License:

| Component              | License      | Compatible |
| ---------------------- | ------------ | ---------- |
| Facebook quaternion.py | BSD-3-Clause | ✓          |
| Alibaba WAN            | Apache 2.0   | ✓          |
| HumanML3D metrics      | MIT          | ✓          |

---

## Legal Notice

-   The Apache 2.0 License of this project applies **only** to original code by Shanda AI Research Tokyo
-   Third-party code remains under its original license
-   All copyright notices are preserved in source files
-   Modifications to third-party code are documented in file headers

For questions, open an issue in the project repository.
