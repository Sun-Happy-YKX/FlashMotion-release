from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT / "eval" / "t2m_ablation_periodic"
OUTPUT = ROOT / "eval" / "t2m_factorial_summary"
STEPS = (1, 2, 4)


def eval_variant(fusion: str, selection: bool, gate: bool, floor: bool) -> str:
    # Preserve the names of the four experiments that predate the factorial.
    legacy = {
        ("add", False, False, False): "tr_only",
        ("adain", False, False, False): "tr_only_adain",
        ("adain", True, False, False): "tr_only_adain_condsel",
        ("adain", True, True, False): "tr_only_adain_condsel_condgate",
    }
    key = (fusion, selection, gate, floor)
    if key in legacy:
        return legacy[key]
    return "t2m_factorial_{}_{}_{}_{}".format(
        fusion,
        "sel" if selection else "nosel",
        "condgate" if gate else "sum",
        "floor1" if floor else "nofloor",
    )


def checkpoint_score(metrics: dict) -> tuple[float, float]:
    rows = [metrics["tables"]["mixed"][f"{step}step"] for step in STEPS]
    fids = [float(row["FID_episode_balanced"]) for row in rows]
    geometric_fid = math.exp(sum(math.log(max(fid, 1e-12)) for fid in fids) / len(fids))
    recall = sum(
        float(row[f"R@{rank}_episode_macro"])
        for row in rows
        for rank in (1, 2, 3)
    ) / 9.0
    # Lower FID is primary. Higher aggregate recall breaks effectively equal ties.
    return geometric_fid, -recall


def main() -> None:
    records = []
    missing = []
    for fusion in ("add", "adain"):
        for selection in (False, True):
            for gate in (False, True):
                for floor in (False, True):
                    variant = eval_variant(fusion, selection, gate, floor)
                    paths = sorted((EVAL_ROOT / variant).glob("step-*/metrics.json"))
                    if not paths:
                        missing.append(variant)
                        continue
                    candidates = [json.loads(path.read_text()) for path in paths]
                    best = min(candidates, key=checkpoint_score)
                    records.append(
                        {
                            "fusion": fusion,
                            "condition_selection": selection,
                            "direction_fusion": "cond_gate" if gate else "sum",
                            "floor_loss": floor,
                            "eval_variant": variant,
                            "evaluated_checkpoints": len(candidates),
                            "best_checkpoint_step": best["checkpoint_step"],
                            "best_checkpoint_epoch": best["checkpoint_epoch"],
                            "selection_geomean_fid": checkpoint_score(best)[0],
                            "metrics": best["tables"]["mixed"],
                        }
                    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_rule": (
            "one checkpoint per setting: minimize the geometric mean of mixed-cohort "
            "FID_episode_balanced over 1/2/4 steps; break ties by mean R@1/R@2/R@3"
        ),
        "expected_settings": 16,
        "available_settings": len(records),
        "missing_eval_variants": missing,
        "settings": records,
    }
    (OUTPUT / "current_best.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# T2M tr_only factorial checkpoint sweep",
        "",
        payload["selection_rule"],
        "",
        f"Available settings: {len(records)}/16.",
        "",
    ]
    for step in STEPS:
        lines.extend(
            [
                f"## {step}-step (same aggregate-selected checkpoint)",
                "",
                "| Fusion | Selection | Direction | Floor | Best step | Epoch | #ckpts | R@1 | R@2 | R@3 | FID | MM-Dist | Div. | PJ gap | AUJ gap |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in sorted(
            records,
            key=lambda item: (
                item["fusion"],
                item["condition_selection"],
                item["direction_fusion"],
                item["floor_loss"],
            ),
        ):
            row = record["metrics"][f"{step}step"]
            transition = row["transition"]
            lines.append(
                "| {fusion} | {selection} | {direction} | {floor} | {ckpt} | {epoch:.1f} | {count} | "
                "{r1:.3f} | {r2:.3f} | {r3:.3f} | {fid:.3f} | {mm:.3f} | {div:.3f} | {pj:.5f} | {auj:.5f} |".format(
                    fusion=record["fusion"],
                    selection="on" if record["condition_selection"] else "off",
                    direction=record["direction_fusion"],
                    floor="on" if record["floor_loss"] else "off",
                    ckpt=record["best_checkpoint_step"],
                    epoch=record["best_checkpoint_epoch"],
                    count=record["evaluated_checkpoints"],
                    r1=row["R@1_episode_macro"],
                    r2=row["R@2_episode_macro"],
                    r3=row["R@3_episode_macro"],
                    fid=row["FID_episode_balanced"],
                    mm=row["MM_Dist_multi_positive"],
                    div=row["Diversity"],
                    pj=transition["PJ_gap"],
                    auj=transition["AUJ_gap_to_cohort_GT"],
                )
            )
        lines.append("")
    if missing:
        lines.extend(["## Missing evaluations", "", *[f"- {item}" for item in missing], ""])
    (OUTPUT / "current_best.md").write_text("\n".join(lines))
    print(f"wrote {OUTPUT / 'current_best.json'}")
    print(f"wrote {OUTPUT / 'current_best.md'}")
    print(f"available={len(records)}/16 missing={len(missing)}")


if __name__ == "__main__":
    main()
