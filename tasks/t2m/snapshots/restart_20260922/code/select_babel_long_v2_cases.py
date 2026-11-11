from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


CATEGORY_PATTERNS = {
    "locomotion": r"\b(walk|run|jog|step|pace|move|locomot)",
    "turning": r"\b(turn|rotate|spin|pivot)",
    "sit_stand": r"\b(sit|stand|rise|get up|lie|lay|kneel|squat)",
    "jump_fall": r"\b(jump|hop|leap|fall|stumble)",
    "upper_body": r"\b(wave|clap|punch|throw|catch|reach|point|hand|arm|gesture)",
    "object_interaction": r"\b(item|object|pick|put|place|carry|lift|open|close|drink)",
    "dance_exercise": r"\b(dance|exercise|stretch|yoga|kick|sports?)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select prediction-blind, duration-stratified BABEL-long cases."
    )
    parser.add_argument(
        "--gt-root", type=Path, default=Path("eval/babel_long_v2/motions/gt")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/babel_long_v2/visualization/selected_20.json"),
    )
    parser.add_argument("--count", type=int, default=20)
    return parser.parse_args()


def words(captions: list[str]) -> set[str]:
    return {
        token
        for caption in captions
        for token in re.findall(r"[a-z]+", caption.lower())
        if len(token) > 2
    }


def categories(captions: list[str]) -> list[str]:
    text = " | ".join(captions).lower()
    return [name for name, pattern in CATEGORY_PATTERNS.items() if re.search(pattern, text)]


def main() -> None:
    args = parse_args()
    if args.count != 20:
        raise ValueError("This benchmark visualization protocol fixes count=20.")
    records = []
    for path in sorted(args.gt_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            captions = [str(value) for value in archive["captions"].tolist()]
            starts = [int(value) for value in archive["segment_starts"].tolist()]
            ends = [int(value) for value in archive["segment_ends"].tolist()]
            length = int(metadata["length"])
        records.append(
            {
                "episode_id": str(metadata["episode_id"]),
                "archive_file": path.name,
                "length_frames": length,
                "duration_seconds": length / 20.0,
                "captions": captions,
                "segment_starts": starts,
                "segment_ends": ends,
                "categories": categories(captions),
                "words": sorted(words(captions)),
            }
        )
    if len(records) < args.count:
        raise RuntimeError(f"Only {len(records)} GT archives were found")

    records.sort(key=lambda row: (row["length_frames"], row["episode_id"]))
    word_frequency = Counter(word for row in records for word in row["words"])
    chosen = []
    used_words: set[str] = set()
    used_categories: Counter[str] = Counter()
    per_bin = args.count // 4
    boundaries = [0, len(records) // 4, len(records) // 2, 3 * len(records) // 4, len(records)]

    for bin_index in range(4):
        candidates = records[boundaries[bin_index] : boundaries[bin_index + 1]]
        for _ in range(per_bin):
            def score(row: dict) -> tuple[float, int, str]:
                unseen_category = sum(1 for cat in row["categories"] if used_categories[cat] == 0)
                category_balance = sum(1.0 / (1 + used_categories[cat]) for cat in row["categories"])
                unseen_words = sum(1.0 / word_frequency[word] for word in row["words"] if word not in used_words)
                segment_bonus = min(len(row["captions"]), 5) * 0.15
                total = 5.0 * unseen_category + category_balance + unseen_words + segment_bonus
                return (total, len(row["captions"]), row["episode_id"])

            winner = max(candidates, key=score)
            candidates = [row for row in candidates if row is not winner]
            winner = dict(winner)
            winner["duration_bin"] = bin_index + 1
            chosen.append(winner)
            used_words.update(winner.pop("words"))
            used_categories.update(winner["categories"])

    chosen.sort(key=lambda row: (row["duration_bin"], row["length_frames"], row["episode_id"]))
    payload = {
        "protocol": "babel_long_v2_visualization_20",
        "selection_is_prediction_blind": True,
        "selection_source": "GT archive metadata and captions only",
        "selection_strategy": (
            "Five episodes from each duration quartile; deterministic greedy coverage of "
            "action categories and rare caption vocabulary within each quartile."
        ),
        "cohort_size": len(records),
        "fps": 20,
        "count": len(chosen),
        "cases": chosen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "duration_bin", "episode_id", "length_frames", "duration_seconds",
                "segments", "categories", "captions",
            ],
        )
        writer.writeheader()
        for case in chosen:
            writer.writerow(
                {
                    "duration_bin": case["duration_bin"],
                    "episode_id": case["episode_id"],
                    "length_frames": case["length_frames"],
                    "duration_seconds": f"{case['duration_seconds']:.2f}",
                    "segments": len(case["captions"]),
                    "categories": "; ".join(case["categories"]),
                    "captions": " | ".join(case["captions"]),
                }
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
