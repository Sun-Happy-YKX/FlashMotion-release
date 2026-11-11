from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectiveSegment:
    annotation_index: int
    start: int
    end: int


def effective_segments(text_ends: list[int], length: int) -> list[EffectiveSegment]:
    """Return the text segments that actually reach the cropped motion.

    BABEL streaming follows FloodDiffusion's convention: ordered ``to_tag``
    values divide the sequence and ``from_tag`` is ignored.  The final caption
    fills any tail after its annotated end.  An annotation whose boundary is
    already beyond the crop therefore does not create a visible segment.
    """

    if not text_ends:
        raise ValueError("BABEL sample has no text boundaries")
    length = int(length)
    previous = 0
    result: list[EffectiveSegment] = []
    for annotation_index, source_end in enumerate(text_ends):
        end = max(previous, min(length, int(source_end)))
        if end > previous:
            result.append(EffectiveSegment(annotation_index, previous, end))
        previous = end
    if previous < length:
        last_index = len(text_ends) - 1
        if result and result[-1].annotation_index == last_index:
            last = result[-1]
            result[-1] = EffectiveSegment(last.annotation_index, last.start, length)
        else:
            result.append(EffectiveSegment(last_index, previous, length))
    return result


def transition_jerk_curve(joints: np.ndarray) -> np.ndarray:
    """FlowMDM jerk: L1 coordinates, then maximum joint, per frame."""

    velocity = joints[1:] - joints[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]
    return np.abs(jerk).sum(axis=-1).max(axis=-1)


def pj_auj(
    transition_curves: list[np.ndarray],
    average_ground_truth_jerk: float = 0.033031363,
) -> dict[str, float]:
    """Official FlowMDM aggregation used by FloodDiffusion's transition metrics."""

    if not transition_curves:
        return {"transitions": 0, "PJ": float("nan"), "AUJ": float("nan")}
    curves = np.stack(transition_curves)
    mean_curve = curves.mean(axis=0)
    return {
        "transitions": int(len(curves)),
        "PJ": float(mean_curve.max()),
        "AUJ": float(np.abs(mean_curve - average_ground_truth_jerk).sum()),
    }
