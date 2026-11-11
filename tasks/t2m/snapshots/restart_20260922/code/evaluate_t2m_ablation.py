from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from t2m_long.babel_long_benchmark import (
    MultiPredictionEmbeddingCollector,
    TRANSITION_FRAMES,
    centered_transition,
    save_shard,
    score_embedding_subset,
    stable_seed,
    transition_metrics,
)
from t2m_long.babel_protocol import effective_segments, pj_auj
from t2m_long.data import MotionTextDataset
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.model import OneShotT2MDecoder
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one FlowerMotion ablation on the no-cap BABEL protocol."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument(
        "--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Deterministic evenly-spaced episode subset; 0 evaluates all BABEL val episodes.",
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        help=(
            "Optional directory whose NAME--HASH.npz files define the exact "
            "BABEL episode cohort. This keeps checkpoint sweeps aligned with "
            "the frozen long-sequence comparison benchmark."
        ),
    )
    return parser.parse_args()


def checkpoint_variant(checkpoint: dict) -> str:
    metadata = checkpoint.get("metadata", {})
    variant = metadata.get("x0_variant")
    if variant:
        return str(variant)
    objective = str(metadata.get("objective", ""))
    for candidate in (
        "x0_flow",
        "diffusion",
        "tr_t0_aux_cfg_baked",
        "tr_t0_aux_rec_only",
        "tr_t0_full_avg",
        "tr_t0_full_rec_only",
        "tr_t0_full_fm_only",
        "tr_cfg_baked",
        "tr_t0_aux",
        "tr_t0_full",
        "t0_only",
        "tr_only",
    ):
        if candidate in objective:
            return candidate
    return "tr_t0_full"


def load_objective(
    checkpoint_path: Path, data_root: Path, device: torch.device
) -> tuple[T2MMeanFlow, dict, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint_variant(checkpoint)
    cross_modal_fusion = str(
        checkpoint.get("metadata", {}).get("cross_modal_fusion", "add")
    )
    condition_selection = bool(
        checkpoint.get("metadata", {}).get("condition_selection", False)
    )
    direction_fusion = str(
        checkpoint.get("metadata", {}).get("direction_fusion", "sum")
    )
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy"))
    objective = T2MMeanFlow(
        OneShotT2MDecoder(
            enable_unconditional=variant
            in {"tr_cfg_baked", "tr_t0_aux_cfg_baked"},
            cross_modal_fusion=cross_modal_fusion,
            condition_selection=condition_selection,
            direction_fusion=direction_fusion,
        ),
        mean,
        std,
        x0_variant=variant,
    ).to(device)
    objective.load_state_dict(checkpoint["model"])
    if "ema_decoder" in checkpoint:
        objective.decoder.load_state_dict(checkpoint["ema_decoder"])
    return objective.eval().requires_grad_(False), checkpoint, variant


@torch.inference_mode()
def sample_episode(
    objective: T2MMeanFlow,
    condition: torch.Tensor,
    steps: int,
    seed: int,
) -> torch.Tensor:
    device = next(objective.parameters()).device
    condition = condition[None].to(device, non_blocking=True)
    mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    normalized = objective.sample(condition, mask, steps, generator)[0]
    return normalized * objective.motion_std[0] + objective.motion_mean[0]


def workload_view(
    arrays: dict[str, np.ndarray], sampling_steps: int, workload: str
) -> tuple[dict[str, np.ndarray], str, np.ndarray]:
    valid = arrays["segment_length"] >= 4
    if workload != "mixed":
        return arrays, f"prediction__{workload}_{sampling_steps}step", valid
    fields = ("ground_truth", "text", "caption", "episode_id")
    view = {field: np.concatenate([arrays[field], arrays[field]]) for field in fields}
    prediction_key = f"prediction__mixed_{sampling_steps}step"
    view[prediction_key] = np.concatenate(
        [
            arrays[f"prediction__single_{sampling_steps}step"],
            arrays[f"prediction__multi_{sampling_steps}step"],
        ]
    )
    return view, prediction_key, np.concatenate([valid, valid])


def score(
    arrays: dict[str, np.ndarray], steps: list[int], device: torch.device, seed: int
) -> dict:
    tables: dict[str, dict] = {}
    for workload in ("single", "multi", "mixed"):
        tables[workload] = {}
        for sampling_steps in steps:
            name = f"{sampling_steps}step"
            view, prediction_key, selected = workload_view(
                arrays, sampling_steps, workload
            )
            metrics = score_embedding_subset(
                view,
                prediction_key,
                selected,
                device,
                seed=seed,
            )
            if workload in {"multi", "mixed"}:
                transition_name = f"multi_{sampling_steps}step"
                transition = transition_metrics(arrays, transition_name)
                official = pj_auj(
                    list(arrays[f"transition__{transition_name}"])
                )
                official_gt = pj_auj(list(arrays["transition__ground_truth"]))
                transition.update(
                    {
                        "AUJ": official["AUJ"],
                        "gt_AUJ": official_gt["AUJ"],
                    }
                )
                metrics["transition"] = transition
            else:
                metrics["transition"] = {
                    "transitions": 0,
                    "PJ": None,
                    "AUJ": None,
                    "AUJ_gap_to_cohort_GT": None,
                }
            tables[workload][name] = metrics
    return tables


def write_markdown(path: Path, record: dict) -> None:
    columns = [
        ("R@1", "R@1_episode_macro"),
        ("R@2", "R@2_episode_macro"),
        ("R@3", "R@3_episode_macro"),
        ("FID", "FID_episode_balanced"),
        ("MM-Dist", "MM_Dist_multi_positive"),
        ("Diversity", "Diversity"),
    ]
    lines = [
        f"# FlowerMotion T2M ablation: {record['variant']}",
        "",
        f"Checkpoint: `{record['checkpoint']}`; BABEL val episodes: {record['episodes']}.",
        "",
    ]
    for split_name, table in record["tables"].items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| Steps | " + " | ".join(name for name, _ in columns) + " | PJ | AUJ | AUJ gap |",
                "|---:|" + "|".join("---:" for _ in range(len(columns) + 3)) + "|",
            ]
        )
        for step_name, metrics in table.items():
            transition = metrics["transition"]
            values = [f"{metrics[key]:.4f}" for _, key in columns]
            pj = transition.get("PJ")
            auj = transition.get("AUJ")
            auj_gap = transition.get("AUJ_gap_to_cohort_GT")
            values.extend(
                [
                    "-" if pj is None else f"{pj:.4f}",
                    "-" if auj is None else f"{auj:.4f}",
                    "-" if auj_gap is None else f"{auj_gap:.4f}",
                ]
            )
            lines.append(f"| {step_name} | " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not args.steps or any(step < 1 for step in args.steps):
        raise ValueError("--steps must contain positive integers")
    root = Path(__file__).resolve().parent
    checkpoint_path = (root / args.checkpoint).resolve()
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    dataset = MotionTextDataset(
        data_root, root / args.embedding_root, "babel", "val", max_length=None
    )
    if args.cohort_dir is not None:
        cohort_dir = (root / args.cohort_dir).resolve()
        cohort_ids = {
            path.name.split("--", 1)[0]
            for path in cohort_dir.glob("*.npz")
            if "--" in path.name
        }
        if not cohort_ids:
            raise RuntimeError(f"No NAME--HASH.npz cohort entries in {cohort_dir}")
        index_by_id = {name: index for index, name in enumerate(dataset.ids)}
        missing = sorted(cohort_ids - index_by_id.keys())
        if missing:
            raise RuntimeError(
                f"Cohort has {len(missing)} IDs absent from BABEL val: {missing[:5]}"
            )
        indices = [index_by_id[name] for name in sorted(cohort_ids)]
    else:
        indices = list(range(len(dataset)))
    if args.limit > 0 and args.limit < len(indices):
        positions = np.linspace(0, len(indices) - 1, args.limit, dtype=np.int64)
        indices = [indices[int(position)] for position in positions]
    objective, checkpoint, variant = load_objective(checkpoint_path, data_root, device)
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    prediction_names = [
        f"{workload}_{step}step"
        for workload in ("single", "multi")
        for step in args.steps
    ]
    collector = MultiPredictionEmbeddingCollector(
        evaluator, prediction_names, args.embedding_batch_size
    )
    transition_parts: dict[str, list[np.ndarray]] = defaultdict(list)

    for dataset_index in tqdm(indices, desc=f"{variant} BABEL no-cap"):
        item = dataset[dataset_index]
        name = item["name"]
        length = int(item["length"])
        segments = effective_segments(item["text_ends"], length)
        ground_truth = (
            item["motion"].to(device) * objective.motion_std[0] + objective.motion_mean[0]
        )
        full_predictions = {
            step: sample_episode(
                objective,
                item["condition"],
                step,
                stable_seed(args.seed, variant, name, step),
            )[:length]
            for step in args.steps
        }

        ground_truth_joints = recover_joints_from_ric(ground_truth[None])[0]
        prediction_joints = {
            step: recover_joints_from_ric(value[None])[0]
            for step, value in full_predictions.items()
        }
        for transition_index, segment in enumerate(segments[1:]):
            gt_curve = centered_transition(
                ground_truth_joints, segment.start, TRANSITION_FRAMES
            )
            if gt_curve is None:
                continue
            transition_parts["transition__ground_truth"].append(gt_curve)
            transition_parts["transition_key"].append(
                np.asarray(f"{name}::{transition_index}", dtype=np.str_)
            )
            transition_parts["transition_episode_id"].append(
                np.asarray(name, dtype=np.str_)
            )
            for step in args.steps:
                curve = centered_transition(
                    prediction_joints[step], segment.start, TRANSITION_FRAMES
                )
                if curve is None:
                    raise RuntimeError("Prediction transition unexpectedly missing")
                transition_parts[f"transition__multi_{step}step"].append(curve)

        for segment_index, segment in enumerate(segments):
            standalone_condition = item["condition"][segment.start : segment.end]
            standalone_predictions = {
                step: sample_episode(
                    objective,
                    standalone_condition,
                    step,
                    stable_seed(args.seed, variant, name, segment_index, "single", step),
                )
                for step in args.steps
            }
            collector.add(
                ground_truth[segment.start : segment.end],
                item["text_tokens"][segment.annotation_index],
                {
                    **{
                        f"single_{step}step": standalone_predictions[step]
                        for step in args.steps
                    },
                    **{
                        f"multi_{step}step": full_predictions[step][
                            segment.start : segment.end
                        ]
                        for step in args.steps
                    },
                },
                {
                    "segment_key": f"{name}::{segment_index}",
                    "episode_id": name,
                    "caption": item["captions"][segment.annotation_index],
                    "segment_index": segment_index,
                    "segment_length": segment.end - segment.start,
                    "episode_length": length,
                },
            )

    arrays = collector.finish()
    for key, values in transition_parts.items():
        arrays[key] = np.stack(values) if values else np.empty((0,), dtype=np.float32)
    shard_path = output / "shard_00_of_01.npz"
    save_shard(
        shard_path,
        arrays,
        {
            "variant": variant,
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint.get("step", 0)),
            "checkpoint_epoch": float(checkpoint.get("epoch", 0.0)),
            "steps": args.steps,
            "split": "FloodDiffusion BABEL_streamed val_processed",
            "episode_max_length": None,
            "episode_sampling": "one-shot whole sequence; no sliding window",
            "semantic_aggregation": "segments scored, then episode macro averaged",
            "episodes": len(indices),
            "segments": len(arrays["segment_key"]),
            "transitions": len(arrays.get("transition_key", [])),
        },
    )
    record = {
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "checkpoint_epoch": float(checkpoint.get("epoch", 0.0)),
        "episodes": len(indices),
        "full_validation_size": len(dataset),
        "periodic_subset": len(indices) != len(dataset),
        "protocol": {
            "upper_length_cap": None,
            "cohort_dir": None
            if args.cohort_dir is None
            else str((root / args.cohort_dir).resolve()),
            "single": "native BABEL episodes with one effective text segment",
            "multi": "native BABEL episodes with at least two effective text segments",
            "mixed": "union of native single- and multi-text episodes",
            "semantic": "each text-aligned segment scored, then macro averaged by episode",
            "transition": "PJ/AUJ on valid multi-text boundaries only",
        },
        "tables": score(arrays, args.steps, device, args.seed),
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    write_markdown(output / "metrics.md", record)
    print(output / "metrics.md", flush=True)


if __name__ == "__main__":
    main()
