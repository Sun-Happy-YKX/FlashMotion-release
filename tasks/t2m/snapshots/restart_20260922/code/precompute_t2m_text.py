from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute one UMT5-XXL sentence feature for every HML3D/BABEL caption."
    )
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--output", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-len", type=int, default=128)
    return parser.parse_args()


def captions_from_file(path: Path) -> list[str]:
    captions: list[str] = []
    for raw in path.read_text().splitlines():
        if raw.strip():
            captions.append(raw.split("#", 1)[0])
    return captions


def collect_captions(data_root: Path) -> list[str]:
    captions: set[str] = set()
    sources = (
        data_root / "HumanML3D" / "texts",
        data_root / "BABEL_streamed" / "texts",
    )
    for text_dir in sources:
        files = sorted(text_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No annotations under {text_dir}")
        for path in tqdm(files, desc=f"scan {text_dir.parent.name}"):
            captions.update(captions_from_file(path))
    # Stable ordering makes the bank reproducible and safe to resume/rebuild.
    return sorted(captions)


def main() -> None:
    args = parse_args()
    flood_root = args.flood_root.resolve()
    data_root = flood_root / "raw_data"
    args.output.mkdir(parents=True, exist_ok=True)
    final_embeddings = args.output / "embeddings.npy"
    final_index = args.output / "text_index.json"
    if final_embeddings.exists() and final_index.exists():
        print(f"Embedding bank already exists: {args.output}")
        return

    captions = collect_captions(data_root)
    print(f"Unique captions: {len(captions):,}")

    sys.path.insert(0, str(flood_root))
    from models.tools.t5 import T5EncoderModel

    checkpoint = flood_root / "deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = flood_root / "deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl"
    if not checkpoint.is_file() or not tokenizer.is_dir():
        raise FileNotFoundError("FloodDiffusion UMT5-XXL checkpoint/tokenizer is incomplete")

    device = torch.device(args.device)
    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
    )

    temporary = args.output / "embeddings.partial.npy"
    bank = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(captions), 4096)
    )
    with torch.inference_mode():
        for start in tqdm(range(0, len(captions), args.batch_size), desc="encode UMT5"):
            batch = captions[start : start + args.batch_size]
            token_features = encoder(batch, device)
            # User-requested frame-wise conditioning uses one semantic vector per
            # caption, repeated over every frame governed by that caption.
            pooled = torch.stack([feature.float().mean(dim=0) for feature in token_features])
            bank[start : start + len(batch)] = pooled.cpu().numpy().astype(np.float16)
    bank.flush()
    del bank

    index = {caption: index for index, caption in enumerate(captions)}
    index_tmp = args.output / "text_index.partial.json"
    index_tmp.write_text(json.dumps(index, ensure_ascii=False))
    os.replace(temporary, final_embeddings)
    os.replace(index_tmp, final_index)
    metadata = {
        "encoder": "UMT5-XXL from FloodDiffusion",
        "pooling": "mean over non-padding tokens",
        "dtype": "float16",
        "dimension": 4096,
        "captions": len(captions),
        "text_len": args.text_len,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(captions):,} embeddings to {args.output}")


if __name__ == "__main__":
    main()
