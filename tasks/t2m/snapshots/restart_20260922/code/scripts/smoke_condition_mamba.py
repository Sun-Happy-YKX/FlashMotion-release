import torch

from model.model import IntraModalMamba


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selective-scan smoke test")
    torch.manual_seed(17)
    device = torch.device("cuda")
    block = IntraModalMamba(
        32,
        dropout=0.0,
        layer_norm_eps=1e-5,
        condition_selection=True,
        direction_fusion="cond_gate",
    ).to(device)
    x = torch.randn(3, 31, 32, device=device, requires_grad=True)
    condition = torch.randn_like(x)
    lengths = torch.tensor([31, 19, 7], device=device)
    mask = torch.arange(31, device=device)[None, :] < lengths[:, None]
    output = block(x, condition=condition, mask=mask)
    loss = output.square().sum() / mask.sum().clamp_min(1)
    loss.backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert output[~mask].abs().max().item() == 0.0
    print(
        f"masked-condition-mamba-smoke-ok loss={loss.item():.6f} "
        f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
