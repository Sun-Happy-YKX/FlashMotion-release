from pathlib import Path

import numpy as np


root = Path("third_party/FloodDiffusion/raw_data/BABEL_streamed")
ids = [line.strip() for line in (root / "val_processed.txt").read_text().splitlines() if line.strip()]
lengths = np.asarray(
    [np.load(root / "motions" / f"{name}.npy", mmap_mode="r").shape[0] for name in ids]
)
annotations = np.asarray(
    [sum(bool(line.strip()) for line in (root / "texts" / f"{name}.txt").read_text().splitlines()) for name in ids]
)
print(
    {
        "n": len(ids),
        "min": int(lengths.min()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "over_200": int((lengths > 200).sum()),
        "over_400": int((lengths > 400).sum()),
        "single": int((annotations == 1).sum()),
        "multi": int((annotations >= 2).sum()),
        "max_annotations": int(annotations.max()),
    }
)
print(sorted(zip(lengths.tolist(), ids), reverse=True)[:20])
