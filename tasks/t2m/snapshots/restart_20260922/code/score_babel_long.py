from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_long_benchmark import (
    combine_shards,
    score_embedding_subset,
    transition_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flower", type=Path, required=True)
    parser.add_argument("--flood", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval/babel_long_v1/report"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_and_merge(flower: dict[str, np.ndarray], flood: dict[str, np.ndarray]):
    for key in (
        "segment_key", "episode_id", "caption", "segment_index",
        "segment_length", "episode_length",
    ):
        if not np.array_equal(flower[key], flood[key]):
            raise RuntimeError(f"Flower/Flood cohort mismatch in {key}")
    # The same frozen GRU evaluator differs by up to ~4e-4 between independent
    # H200 passes because cuDNN reductions are not bitwise deterministic.
    if not np.allclose(flower["ground_truth"], flood["ground_truth"], atol=1e-3, rtol=1e-3):
        raise RuntimeError("Flower/Flood GT embeddings disagree")
    if not np.allclose(flower["text"], flood["text"], atol=2e-4, rtol=2e-4):
        raise RuntimeError("Flower/Flood text embeddings disagree")
    result = dict(flower)
    for key, value in flood.items():
        if key.startswith("prediction__"):
            result[key] = value
    return result


def workload_view(
    arrays: dict[str, np.ndarray], prediction_names: list[str], output_name: str
) -> tuple[dict[str, np.ndarray], str]:
    if len(prediction_names) == 1:
        return arrays, f"prediction__{prediction_names[0]}"
    segment_fields = (
        "ground_truth", "text", "segment_key", "episode_id", "caption",
        "segment_index", "segment_length", "episode_length",
    )
    view = {
        key: np.concatenate([arrays[key] for _ in prediction_names])
        for key in segment_fields
    }
    prediction_key = f"prediction__{output_name}"
    view[prediction_key] = np.concatenate(
        [arrays[f"prediction__{name}"] for name in prediction_names]
    )
    return view, prediction_key


def score_table(
    arrays: dict[str, np.ndarray],
    models: dict[str, str],
    workload: str,
    selected: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict:
    table = {}
    for model_name, suffix in models.items():
        workloads = [f"{workload}_{suffix}"] if workload != "mixed" else [
            f"single_{suffix}", f"multi_{suffix}"
        ]
        view, prediction_key = workload_view(
            arrays, workloads, f"{model_name}_mixed"
        )
        chosen = selected if workload != "mixed" else np.concatenate([selected, selected])
        table[model_name] = score_embedding_subset(
            view, prediction_key, chosen, device, seed=seed
        )
    return table


def fmt(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None or not np.isfinite(value):
        return "-"
    return f"{float(value):.4f}"


def markdown_table(title: str, table: dict) -> str:
    columns = [
        ("R@1 macro", "R@1_episode_macro"),
        ("R@2 macro", "R@2_episode_macro"),
        ("R@3 macro", "R@3_episode_macro"),
        ("R@1 micro", "R@1_segment_micro"),
        ("FID", "FID_segment"),
        ("FID balanced", "FID_episode_balanced"),
        ("MM-Dist", "MM_Dist_multi_positive"),
        ("Diversity", "Diversity"),
        ("Episodes", "episodes"),
        ("Segments", "segments"),
    ]
    lines = [f"## {title}", "", "| Method | " + " | ".join(x[0] for x in columns) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
    first = next(iter(table.values()))
    gt = {
        "R@1_episode_macro": first.get("gt_R@1_episode_macro"),
        "R@2_episode_macro": first.get("gt_R@2_episode_macro"),
        "R@3_episode_macro": first.get("gt_R@3_episode_macro"),
        "R@1_segment_micro": first.get("gt_R@1_segment_micro"),
        "FID_segment": 0.0,
        "FID_episode_balanced": 0.0,
        "MM_Dist_multi_positive": first.get("gt_MM_Dist_multi_positive"),
        "Diversity": first.get("gt_Diversity"),
        "episodes": first.get("episodes"),
        "segments": first.get("segments"),
    }
    lines.append("| GT | " + " | ".join(fmt(gt.get(key)) for _, key in columns) + " |")
    for name, values in table.items():
        lines.append(
            f"| {name} | " + " | ".join(fmt(values.get(key)) for _, key in columns) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flower_paths = sorted((root / args.flower).resolve().glob("shard_*_of_*.npz"))
    flood_paths = sorted((root / args.flood).resolve().glob("shard_*_of_*.npz"))
    flower = combine_shards(flower_paths)
    flood = combine_shards(flood_paths)
    arrays = validate_and_merge(flower, flood)
    models = {
        "FlowerMotion 1-step": "1step",
        "FlowerMotion 2-step": "2step",
        "FlowerMotion 4-step": "4step",
        "FloodDiffusion 10-step": "10step",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = arrays["segment_length"] >= 4
    long_episode = arrays["episode_length"] > 200
    report = {
        "protocol": {
            "dataset": "FloodDiffusion public BABEL_streamed val_processed",
            "fps": 20,
            "episode_max_length": None,
            "native_episodes": int(len(np.unique(arrays["episode_id"]))),
            "native_annotations_per_episode": 2,
            "single": "each annotated action generated independently",
            "multi": "the complete native two-prompt episode generated in one call",
            "mixed": "equal union of the single and multi workloads",
            "primary_recall": "duplicate-safe multi-positive, episode macro average",
            "transition_window": "20 frames / 1 second centered at each valid boundary",
        },
        "tables": {},
    }
    report["tables"]["multi_all"] = score_table(
        arrays, models, "multi", valid, device, args.seed
    )
    report["tables"]["multi_long_gt_10s"] = score_table(
        arrays, models, "multi", valid & long_episode, device, args.seed
    )
    report["tables"]["single_control"] = score_table(
        arrays, models, "single", valid, device, args.seed
    )
    report["tables"]["mixed"] = score_table(
        arrays, models, "mixed", valid, device, args.seed
    )

    duration_masks = {
        "multi_duration_0_10s": valid & (arrays["episode_length"] <= 200),
        "multi_duration_10_20s": valid & (arrays["episode_length"] > 200)
        & (arrays["episode_length"] <= 400),
        "multi_duration_gt_20s": valid & (arrays["episode_length"] > 400),
    }
    for name, selected in duration_masks.items():
        report["tables"][name] = score_table(
            arrays, models, "multi", selected, device, args.seed
        )
    for minimum in (20, 40):
        report["tables"][f"multi_min_segment_{minimum}f"] = score_table(
            arrays,
            models,
            "multi",
            arrays["segment_length"] >= minimum,
            device,
            args.seed,
        )

    transitions = {}
    for model_name, suffix in models.items():
        source = flower if suffix != "10step" else flood
        transitions[model_name] = transition_metrics(source, f"multi_{suffix}")
    report["tables"]["transitions"] = transitions

    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    sections = [
        "# BABEL Long-Sequence Benchmark v1",
        "",
        "All episodes are generated without a 200-frame cap. Primary Recall is "
        "duplicate-safe, segment-scored and episode-macro averaged.",
        "",
    ]
    titles = {
        "multi_all": "Multi/Long — all complete episodes",
        "multi_long_gt_10s": "Multi/Long — episodes longer than 10 seconds",
        "single_control": "Single-text control",
        "mixed": "Mixed single + multi workloads",
        "multi_duration_0_10s": "Multi — 0–10 seconds",
        "multi_duration_10_20s": "Multi — 10–20 seconds",
        "multi_duration_gt_20s": "Multi — over 20 seconds",
        "multi_min_segment_20f": "Multi — action segments at least 1 second",
        "multi_min_segment_40f": "Multi — action segments at least 2 seconds",
    }
    for key, title in titles.items():
        sections.extend([markdown_table(title, report["tables"][key]), ""])
    transition_lines = [
        "## Transition smoothness (one-second windows)",
        "",
        "| Method | PJ | GT PJ | PJ gap | AUJ gap to cohort GT | Transitions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in transitions.items():
        transition_lines.append(
            f"| {name} | {fmt(values.get('PJ'))} | {fmt(values.get('gt_PJ'))} | "
            f"{fmt(values.get('PJ_gap'))} | {fmt(values.get('AUJ_gap_to_cohort_GT'))} | "
            f"{fmt(values.get('transitions'))} |"
        )
    sections.extend(transition_lines)
    (output / "tables.md").write_text("\n".join(sections))
    print(output / "tables.md")


if __name__ == "__main__":
    main()
