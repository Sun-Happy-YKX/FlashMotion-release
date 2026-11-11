from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


METHODS = [
    {
        "key": "flowermotion_1step",
        "label": "FlowerMotion (1 step)",
        "long_mode": "native one-shot whole episode",
        "source": "tr_t0_full checkpoint step 260k",
        "url": "",
    },
    {
        "key": "flowermotion_2step",
        "label": "FlowerMotion (2 steps)",
        "long_mode": "native one-shot whole episode",
        "source": "tr_t0_full checkpoint step 260k",
        "url": "",
    },
    {
        "key": "flowermotion_4step",
        "label": "FlowerMotion (4 steps)",
        "long_mode": "native one-shot whole episode",
        "source": "tr_t0_full checkpoint step 260k",
        "url": "",
    },
    {
        "key": "flooddiffusion",
        "label": "FloodDiffusion",
        "long_mode": "native one-shot multi-caption",
        "source": "official repository/checkpoint",
        "url": "https://github.com/thu-ml/FloodDiffusion",
    },
    {
        "key": "flowmdm",
        "label": "FlowMDM",
        "long_mode": "native blended positional encoding",
        "source": "official repository/checkpoint",
        "url": "https://github.com/BarqueroGerman/FlowMDM",
    },
    {
        "key": "dart",
        "label": "DART",
        "long_mode": "native autoregressive motion primitives",
        "source": "official repository/checkpoint",
        "url": "https://github.com/zkf1997/DART",
    },
    {
        "key": "motionstreamer",
        "label": "MotionStreamer",
        "long_mode": "native continuous streaming",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/zju3dv/MotionStreamer",
    },
    {
        "key": "doubletake",
        "label": "DoubleTake",
        "long_mode": "native overlapping two-take transitions",
        "source": "official PriorMDM repository/checkpoint",
        "url": "https://github.com/priorMDM/priorMDM",
    },
    {
        "key": "mdm",
        "label": "MDM",
        "long_mode": "segment-conditioned hard concatenation",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/GuyTevet/motion-diffusion-model",
    },
    {
        "key": "momask",
        "label": "MoMask",
        "long_mode": "segment-conditioned hard concatenation",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/EricGuo5513/momask-codes",
    },
    {
        "key": "mld",
        "label": "MLD",
        "long_mode": "segment-conditioned hard concatenation",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/ChenFengYe/motion-latent-diffusion",
    },
    {
        "key": "t2mgpt",
        "label": "T2M-GPT",
        "long_mode": "segment-conditioned concatenation; EOS outputs exact-length padded",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/Mael-zys/T2M-GPT",
    },
    {
        "key": "motiongpt",
        "label": "MotionGPT",
        "long_mode": "segment-conditioned hard concatenation",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/OpenMotionLab/MotionGPT",
    },
    {
        "key": "motionlcm",
        "label": "MotionLCM (1 step)",
        "long_mode": "segment-conditioned hard concatenation",
        "source": "official weights repackaged by Motius",
        "url": "https://github.com/Dai-Wenxun/MotionLCM",
    },
]


def metric_row(spec: dict, score_root: Path) -> dict:
    data = json.loads((score_root / f"{spec['key']}_metrics.json").read_text())
    semantic = data["semantic"]
    transition = data["transition"]
    validation = data["archive_validation"]
    return {
        "method": spec["label"],
        "key": spec["key"],
        "long_mode": spec["long_mode"],
        "source": spec["source"],
        "episodes": validation["episodes"],
        "annotated_segments": validation["segments"],
        "segments": semantic["segments"],
        "transitions": transition["transitions"],
        "R@1": semantic["R@1_episode_macro"],
        "R@2": semantic["R@2_episode_macro"],
        "R@3": semantic["R@3_episode_macro"],
        "FID": semantic["FID_episode_balanced"],
        "MM-Dist": semantic["MM_Dist_multi_positive"],
        "Diversity": semantic["Diversity"],
        "PJ": transition["PJ"],
        "PJ-gap": transition["PJ_gap"],
        "AUJ-gap": transition["AUJ_gap_to_cohort_GT"],
    }


def ground_truth_row(score_root: Path) -> dict:
    data = json.loads((score_root / "flowermotion_4step_metrics.json").read_text())
    semantic = data["semantic"]
    transition = data["transition"]
    validation = data["archive_validation"]
    return {
        "method": "Ground truth",
        "key": "gt",
        "long_mode": "reference motion",
        "source": "FloodDiffusion-processed BABEL validation",
        "episodes": validation["episodes"],
        "annotated_segments": validation["segments"],
        "segments": semantic["segments"],
        "transitions": transition["transitions"],
        "R@1": semantic["gt_R@1_episode_macro"],
        "R@2": semantic["gt_R@2_episode_macro"],
        "R@3": semantic["gt_R@3_episode_macro"],
        "FID": 0.0,
        "MM-Dist": semantic["gt_MM_Dist_multi_positive"],
        "Diversity": semantic["gt_Diversity"],
        "PJ": transition["gt_PJ"],
        "PJ-gap": 0.0,
        "AUJ-gap": 0.0,
    }


def inventory(method_root: Path, method: str) -> list[dict]:
    result = []
    for path in sorted((method_root / method).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            result.append(
                {
                    "episode_id": metadata["episode_id"],
                    "length": int(metadata["length"]),
                    "fps": int(metadata["fps"]),
                    "captions": data["captions"].tolist(),
                    "segment_starts": data["segment_starts"].astype(int).tolist(),
                    "segment_ends": data["segment_ends"].astype(int).tolist(),
                    "archive": str(path.resolve()),
                }
            )
    return result


def main() -> None:
    root = Path(__file__).resolve().parent
    eval_root = root / "eval/babel_long_v2"
    score_root = eval_root / "scores"
    report_root = eval_root / "report"
    motion_root = eval_root / "motions"
    report_root.mkdir(parents=True, exist_ok=True)
    rows = [ground_truth_row(score_root), *[metric_row(spec, score_root) for spec in METHODS]]
    cohort_shapes = {
        (
            row["episodes"],
            row["annotated_segments"],
            row["segments"],
            row["transitions"],
        )
        for row in rows
    }
    if len(cohort_shapes) != 1:
        raise RuntimeError(f"Methods do not share one cohort shape: {sorted(cohort_shapes)}")
    (
        cohort_episodes,
        cohort_annotated_segments,
        cohort_scored_segments,
        cohort_transitions,
    ) = next(iter(cohort_shapes))
    fields = list(rows[0])
    with (report_root / "baseline_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    numeric = {"R@1", "R@2", "R@3", "FID", "MM-Dist", "Diversity", "PJ", "PJ-gap", "AUJ-gap"}
    display_fields = [
        "method", "long_mode", "episodes", "segments", "transitions",
        "R@1", "R@2", "R@3", "FID", "MM-Dist", "Diversity", "PJ", "PJ-gap", "AUJ-gap",
    ]
    lines = [
        "# BABEL Long v2 — FlowerMotion and 11-baseline comparison",
        "",
        "Cohort: official FloodDiffusion-processed BABEL validation, multi-caption episodes longer than 200 frames at 20 fps; no upper length cap. Semantic metrics are computed per annotated segment then macro-averaged per episode. Transition metrics use the shared one-second boundary window.",
        "",
        "Directions: R@k higher is better; FID/MM-Dist/PJ-gap/AUJ-gap lower is better; Diversity and raw PJ should be read against the Ground-truth row.",
        "",
        "| " + " | ".join(display_fields) + " |",
        "| " + " | ".join(["---"] * len(display_fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in display_fields:
            value = row[field]
            values.append(f"{value:.4f}" if field in numeric else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Native-long and segment-adapted methods are explicitly separated by `long_mode`; they must not be presented as identical inference protocols.",
            "T2M-GPT may emit EOS before the requested duration. Its per-run exact-length padding count is retained in the method archive metadata/run summary.",
        ]
    )
    (report_root / "baseline_table.md").write_text("\n".join(lines) + "\n")

    inventories = {
        "gt": inventory(motion_root, "gt"),
        **{spec["key"]: inventory(motion_root, spec["key"]) for spec in METHODS},
    }
    (report_root / "visualization_inventory.json").write_text(
        json.dumps(inventories, indent=2) + "\n"
    )
    manifest = {
        "benchmark": "babel_long_v2_gt10s_no_cap",
        "cohort": {
            "split": "BABEL validation",
            "minimum_frames": 201,
            "fps": 20,
            "upper_length_cap": None,
            "episodes": cohort_episodes,
            "annotated_segments": cohort_annotated_segments,
            "semantic_scored_segments": cohort_scored_segments,
            "semantic_minimum_segment_frames": 4,
            "transitions": cohort_transitions,
        },
        "archive_schema": {
            "format": "compressed NumPy npz",
            "motion_263": "(T,263) denormalized HumanML3D features",
            "joints_y_up": "(T,22,3) absolute Y-up joints",
            "captions": "ordered segment text",
            "segment_starts_and_ends": "contiguous half-open frame intervals",
        },
        "methods": [
            {
                **spec,
                "score": str((score_root / f"{spec['key']}_metrics.json").resolve()),
                "motion_directory": str((motion_root / spec["key"]).resolve()),
                "saved_episodes": len(inventories[spec["key"]]),
            }
            for spec in METHODS
        ],
    }
    (report_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report_root / "baseline_table.md")


if __name__ == "__main__":
    main()
