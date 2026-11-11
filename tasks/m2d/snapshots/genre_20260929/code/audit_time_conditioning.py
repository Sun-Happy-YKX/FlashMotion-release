"""Runtime audit for the (t, t-r) conditioning contract."""

from __future__ import annotations

import torch
from torch import nn

from model.flowmatching import FlowMatching


class SpyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x, cond, genre, time, interval, cond_drop_mask=None):
        self.calls.append((time.detach().cpu(), interval.detach().cpu()))
        return x * self.scale

    def infer_pred(self, x, cond, genre, time, interval):
        return self.forward(x, cond, genre, time, interval)


def rounded_pairs(calls):
    return {
        (round(float(time[0]), 4), round(float(interval[0]), 4))
        for time, interval in calls
    }


def main() -> None:
    decoder = SpyDecoder()
    objective = FlowMatching(
        decoder,
        horizon=5,
        repr_dim=6,
        smpl=None,
        x0_variant="tr_t0_full_avg",
        loss_weights={"rec": 0.0, "vel": 0.0, "acc": 0.0, "fk": 0.0, "floor": 0.0},
    )
    objective.sample_two_timesteps = lambda batch, device: (
        torch.full((batch, 1, 1), 0.8, device=device),
        torch.full((batch, 1, 1), 0.3, device=device),
    )
    motion = torch.randn(2, 5, 6)
    condition = torch.randn(2, 5, 8)
    genre = torch.zeros(2, dtype=torch.long)
    loss, _ = objective(motion, condition, genre)
    loss.backward()
    train_pairs = rounded_pairs(decoder.calls)
    assert (0.8, 0.5) in train_pairs, train_pairs
    assert (0.8, 0.8) in train_pairs, train_pairs

    decoder.calls.clear()
    objective.euler_sample(motion.shape, condition, genre, n_steps=5)
    sample_pairs = rounded_pairs(decoder.calls)
    assert sample_pairs == {
        (1.0, 0.25),
        (0.75, 0.25),
        (0.5, 0.25),
        (0.25, 0.25),
    }, sample_pairs
    print("time_conditioning_ok", "train", sorted(train_pairs), "sample", sorted(sample_pairs))


if __name__ == "__main__":
    main()
