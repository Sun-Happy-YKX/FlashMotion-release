from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from model.adan import Adan
from t2m_long.data import build_joint_train_dataset, collate_motion_text
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.model import OneShotT2MDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-shot FlowerDance on HumanML3D+BABEL.")
    parser.add_argument("--data-root", type=Path, default=Path("third_party/FloodDiffusion/raw_data"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("runs/t2m_long_tr_t0_full"))
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=50,
        help="Checkpoint cadence in data epochs; 0 falls back to --save-every steps.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-t", type=float, default=0.05)
    parser.add_argument("--min-interval", type=float, default=0.05)
    parser.add_argument("--fm-time-weight-power", type=float, default=2.0)
    parser.add_argument("--fm-weight", type=float, default=1.0)
    parser.add_argument("--rec-weight", type=float, default=0.636)
    parser.add_argument("--vel-weight", type=float, default=0.323)
    parser.add_argument("--acc-weight", type=float, default=0.2)
    parser.add_argument("--fk-weight", type=float, default=0.646)
    parser.add_argument(
        "--floor-weight",
        type=float,
        default=0.0,
        help="Weight for recovered-coordinate floor/contact loss (Y-up HumanML3D joints).",
    )
    parser.add_argument(
        "--x0-variant",
        choices=(
            "x0_flow",
            "diffusion",
            "t0_only",
            "tr_only",
            "tr_t0_aux",
            "tr_t0_full",
            "tr_t0_full_avg",
            "tr_cfg_baked",
            "tr_t0_aux_cfg_baked",
            "tr_t0_full_fm_only",
            "tr_t0_full_rec_only",
            "tr_t0_aux_rec_only",
        ),
        default="tr_t0_full",
    )
    parser.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    parser.add_argument("--cfg-omega", type=float, default=2.0)
    parser.add_argument(
        "--cross-modal-fusion",
        choices=("add", "adain"),
        default="add",
        help="Cross-modal injection used in every dance block.",
    )
    parser.add_argument(
        "--condition-selection",
        action="store_true",
        help="Inject the frame-wise text condition directly into Mamba delta/B/C selection parameters.",
    )
    parser.add_argument(
        "--direction-fusion",
        choices=("sum", "cond_gate"),
        default="sum",
        help="Fuse forward/backward Mamba streams by summation or a condition-driven gate.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    return parser.parse_args()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    ddp = DistributedDataParallelKwargs(
        find_unused_parameters=args.x0_variant in {
            "tr_cfg_baked",
            "tr_t0_aux_cfg_baked",
        }
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp])
    set_seed(args.seed, device_specific=True)

    dataset = build_joint_train_dataset(args.data_root, args.embedding_root, args.max_length)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
        collate_fn=collate_motion_text,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    mean = torch.from_numpy(np.load(args.data_root / "HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(args.data_root / "HumanML3D/Std.npy"))
    decoder = OneShotT2MDecoder(
        enable_unconditional=args.x0_variant in {
            "tr_cfg_baked",
            "tr_t0_aux_cfg_baked",
        },
        cross_modal_fusion=args.cross_modal_fusion,
        condition_selection=args.condition_selection,
        direction_fusion=args.direction_fusion,
    )
    objective = T2MMeanFlow(
        decoder,
        mean,
        std,
        min_t=args.min_t,
        min_interval=args.min_interval,
        fm_time_weight_power=args.fm_time_weight_power,
        x0_variant=args.x0_variant,
        cfg_dropout_prob=args.cfg_dropout_prob,
        cfg_omega=args.cfg_omega,
        loss_weights={
            "fm": args.fm_weight,
            "rec": args.rec_weight,
            "vel": args.vel_weight,
            "acc": args.acc_weight,
            "fk": args.fk_weight,
            "floor": args.floor_weight,
        },
    )
    optimizer = Adan(
        objective.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        objective.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    objective, optimizer, loader = accelerator.prepare(objective, optimizer, loader)
    live_objective = accelerator.unwrap_model(objective)
    ema_decoder = copy.deepcopy(live_objective.decoder).eval().requires_grad_(False)
    if checkpoint is not None and "ema_decoder" in checkpoint:
        ema_decoder.load_state_dict(checkpoint["ema_decoder"])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train.jsonl"
    steps_per_epoch = len(loader)
    save_every_steps = (
        args.save_every_epochs * steps_per_epoch
        if args.save_every_epochs > 0
        else args.save_every
    )
    if save_every_steps <= 0:
        raise ValueError("Checkpoint cadence must be positive")
    metadata = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "embedding_root": str(args.embedding_root.resolve()),
        "output": str(args.output.resolve()),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "objective": f"x0 {args.x0_variant}",
        "datasets": ["BABEL_streamed", "HumanML3D"],
        "representation": "standard normalized HumanML3D 263D",
        "height_axis": "Y absolute root height",
        "horizontal_plane": "XZ root local velocities integrated by official recovery",
        "floor_loss": args.floor_weight,
        "sampling": "whole-sequence generation, 1/2/4-step evaluation",
        "eval_protocol": (
            "BABEL val with no upper length cap; segment-scored episode-macro "
            "single/multi/mixed tables and PJ/AUJ at multi-segment boundaries"
        ),
        "world_size": accelerator.num_processes,
        "global_batch_size": args.batch_size * accelerator.num_processes,
        "train_samples": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "checkpoint_every_steps": save_every_steps,
    }
    if accelerator.is_main_process:
        serializable = {
            key: value if isinstance(value, (str, int, float, bool, type(None), list)) else str(value)
            for key, value in metadata.items()
        }
        (args.output / "config.json").write_text(json.dumps(serializable, indent=2))
        print(json.dumps(serializable, indent=2), flush=True)
        print(f"Model parameters: {sum(p.numel() for p in objective.parameters()):,}", flush=True)

    accelerator.wait_for_everyone()
    objective.train()
    data_iterator = iter(loader)
    rolling: dict[str, torch.Tensor] = {}
    rolling_count = 0
    wall_start = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        optimizer.zero_grad(set_to_none=True)
        total, losses = objective(batch["motion"], batch["condition"], batch["mask"])
        accelerator.backward(total)
        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(objective.parameters(), args.max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, live_parameter in zip(
                ema_decoder.parameters(), live_objective.decoder.parameters()
            ):
                ema_parameter.lerp_(live_parameter.detach(), 1.0 - args.ema_decay)

        local_values = {"total": total.detach(), **{name: value.detach() for name, value in losses.items()}}
        for name, value in local_values.items():
            rolling[name] = rolling.get(name, torch.zeros_like(value)) + value
        rolling_count += 1
        if step % args.log_every == 0:
            names = list(rolling)
            local_average = torch.stack([rolling[name] / rolling_count for name in names])
            global_average = accelerator.reduce(local_average, reduction="mean")
            if accelerator.is_main_process:
                elapsed = time.time() - wall_start
                record = {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "seconds_per_step": elapsed / max(1, step - start_step),
                    "rank0_peak_memory_gib": torch.cuda.max_memory_allocated()
                    / (1024**3),
                    **{
                        name: float(value.cpu())
                        for name, value in zip(names, global_average)
                    },
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            rolling.clear()
            rolling_count = 0

        should_save = step % save_every_steps == 0 or step == args.max_steps
        if should_save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(objective)
                state = {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "model": unwrapped.state_dict(),
                    "ema_decoder": ema_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metadata": metadata,
                }
                atomic_save(state, checkpoint_dir / f"step-{step:07d}.pt")
                atomic_save(state, checkpoint_dir / "latest.pt")
                print(f"Checkpoint saved at step {step}", flush=True)
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
