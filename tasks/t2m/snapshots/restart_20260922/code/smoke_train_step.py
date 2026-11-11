"""Run one real training update, optionally under ``accelerate launch``."""

import argparse
import pickle

import torch

from EDGE import EDGE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x0_variant",
        choices=(
            "t0_only",
            "tr_only",
            "tr_t0_aux",
            "tr_t0_full",
            "tr_cfg_baked",
        ),
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dataset",
        default="data/dataset_backups_stable_v2/test_tensor_dataset.pkl",
    )
    args = parser.parse_args()

    with open(args.dataset, "rb") as handle:
        dataset = pickle.load(handle)
    model = EDGE(feature_type="baseline", x0_variant=args.x0_variant)
    model.flow_matching.set_normalizer(dataset.normalizer)
    device = model.accelerator.device
    count = args.batch_size
    rank = model.accelerator.process_index
    start = rank * count
    stop = start + count
    x = dataset.data["pose"][start:stop].to(device)
    cond = dataset.data["features"][start:stop].to(device)
    genre = torch.as_tensor(
        dataset.data["genres"][start:stop], dtype=torch.long, device=device
    )

    # Distinct data and stochastic paths per rank make this a real DDP sync
    # check: an accidentally bypassed wrapper can no longer pass by coincidence.
    torch.manual_seed(1234 + rank)
    model.optim.zero_grad()
    total, losses = model.flow_matching(x, cond, genre)
    model.accelerator.backward(total)
    model.accelerator.clip_grad_norm_(model.model.parameters(), 1.0)
    model.optim.step()
    model.accelerator.wait_for_everyone()
    gathered = model.accelerator.gather(total.detach().reshape(1))
    first_parameter = (
        model.accelerator.unwrap_model(model.model)
        .input_projection.weight.detach()
        .float()
    )
    checksum = torch.stack(
        (first_parameter.sum(), first_parameter.square().sum())
    )[None]
    gathered_checksums = model.accelerator.gather(checksum)
    if device.type == "cuda":
        memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            ],
            device=device,
        )[None]
    else:
        memory = torch.zeros((1, 2), device=device)
    gathered_memory = model.accelerator.gather(memory)
    synced = torch.allclose(
        gathered_checksums,
        gathered_checksums[:1].expand_as(gathered_checksums),
        rtol=1e-6,
        atol=1e-7,
    )
    if not synced:
        raise RuntimeError(
            f"DDP parameters diverged after one update: {gathered_checksums.cpu()}"
        )
    if model.accelerator.is_main_process:
        labels = ("fm", "rec", "vel", "acc", "fk", "floor")
        print(
            "distributed_training_step_ok",
            args.x0_variant,
            "world_size",
            model.accelerator.num_processes,
            "losses",
            dict(zip(labels, [float(value.detach().cpu()) for value in losses])),
            "rank_totals",
            gathered.cpu().tolist(),
            "parameter_checksums",
            gathered_checksums.cpu().tolist(),
            "max_memory_gib_allocated_reserved",
            gathered_memory.cpu().tolist(),
            flush=True,
        )
    model.accelerator.end_training()


if __name__ == "__main__":
    main()
