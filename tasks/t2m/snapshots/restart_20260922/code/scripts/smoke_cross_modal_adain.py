import torch

from model.model import CrossModalAdaIN
from t2m_long.model import OneShotT2MDecoder


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(7)

    adain = CrossModalAdaIN(32).to(device)
    content = torch.randn(3, 17, 32, device=device, requires_grad=True)
    condition = torch.randn_like(content)
    mask = torch.arange(17, device=device)[None] < torch.tensor([17, 13, 9], device=device)[:, None]
    initial = adain(content, condition, mask=mask)
    identity_error = ((initial - content) * mask[..., None]).abs().max().item()
    padded_error = (initial * ~mask[..., None]).abs().max().item()
    assert identity_error < 2e-5, identity_error
    assert padded_error == 0.0, padded_error

    model = OneShotT2MDecoder(
        motion_dim=263,
        text_dim=64,
        latent_dim=32,
        dropout=0.0,
        cross_modal_fusion="adain",
    ).to(device)
    motion = torch.randn(2, 17, 263, device=device)
    cond = torch.randn(2, 17, 64, device=device)
    batch_mask = torch.arange(17, device=device)[None] < torch.tensor([17, 11], device=device)[:, None]
    time = torch.tensor([0.8, 0.5], device=device)
    interval = torch.tensor([0.3, 0.2], device=device)
    output = model(motion, cond, batch_mask, time, interval)
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert model.decoder.dancelayerstack[0].cross_adain.modulation.weight.grad is not None
    assert (output * ~batch_mask[..., None]).abs().max().item() == 0.0
    print(
        {
            "identity_error": identity_error,
            "padded_error": padded_error,
            "loss": loss.item(),
            "shape": tuple(output.shape),
        }
    )


if __name__ == "__main__":
    main()
