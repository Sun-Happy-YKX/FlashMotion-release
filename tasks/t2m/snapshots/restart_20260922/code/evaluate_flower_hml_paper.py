from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from t2m_long.data import TextEmbeddingStore, _parse_text
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.model import OneShotT2MDecoder


ROOT = Path('/efs/yangkaixing/FlowerDance_T2M_Long')
FLOOD_ROOT = ROOT / 'third_party/FloodDiffusion'


def load_flood_caption_protocol(folder: Path) -> list[tuple[int, str]]:
    keys = []
    captions = []
    for path in sorted(folder.glob('shard_*.npz')):
        with np.load(path) as shard:
            keys.append(shard['keys'].copy())
            captions.append(shard['captions'].copy())
    if not keys:
        raise RuntimeError(f'No Flood protocol shards found in {folder}')
    keys = np.concatenate(keys)
    captions = np.concatenate(captions)
    order = np.argsort(keys, kind='stable')
    return [(int(keys[index]), str(captions[index])) for index in order]


class HumanMLMatchedDataset(Dataset):
    """The exact valid IDs and captions used by the FloodDiffusion reproduction."""

    def __init__(self, embedding_root: Path, flood_protocol: Path):
        hml = FLOOD_ROOT / 'raw_data/HumanML3D'
        self.motion_root = hml / 'new_joint_vecs'
        self.text_root = hml / 'texts'
        self.names = [line.strip() for line in (hml / 'test.txt').read_text().splitlines() if line.strip()]
        self.items = load_flood_caption_protocol(flood_protocol)
        self.embedding_store = TextEmbeddingStore(embedding_root)
        self.mean = np.load(hml / 'Mean.npy').astype(np.float32)
        self.std = np.load(hml / 'Std.npy').astype(np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        key, caption = self.items[index]
        name = self.names[key]
        motion = np.load(self.motion_root / f'{name}.npy').astype(np.float32)
        if (
            motion.ndim != 2
            or motion.shape[1] != 263
            or len(motion) > 200
            or not np.isfinite(motion).all()
        ):
            raise RuntimeError(f'Flood protocol contains invalid sample: key={key} name={name}')
        annotations = _parse_text(self.text_root / f'{name}.txt')
        matches = [item for item in annotations if item[0] == caption]
        if not matches:
            raise RuntimeError(f'Caption not found for key={key} name={name}: {caption!r}')
        tokens = matches[0][1]
        length = len(motion)
        condition = np.repeat(
            self.embedding_store.get(caption)[None].astype(np.float16), length, axis=0
        )
        normalized = (motion - self.mean) / self.std
        return {
            'motion': torch.from_numpy(normalized),
            'condition': torch.from_numpy(condition),
            'length': length,
            'tokens': tokens,
            'caption': caption,
            'key': key,
            'name': name,
        }


def collate(items: list[dict]) -> dict:
    motions = pad_sequence([item['motion'] for item in items], batch_first=True)
    conditions = pad_sequence([item['condition'] for item in items], batch_first=True)
    motions = torch.nn.functional.pad(motions, (0, 0, 0, 200 - motions.shape[1]))
    conditions = torch.nn.functional.pad(conditions, (0, 0, 0, 200 - conditions.shape[1]))
    lengths = torch.tensor([item['length'] for item in items], dtype=torch.long)
    mask = torch.arange(200)[None] < lengths[:, None]
    return {
        'motion': motions,
        'condition': conditions,
        'lengths': lengths,
        'mask': mask,
        'tokens': [item['tokens'] for item in items],
        'captions': [item['caption'] for item in items],
        'keys': np.asarray([item['key'] for item in items], dtype=np.int64),
        'names': [item['name'] for item in items],
    }


def metric_repeats(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    text: torch.Tensor,
    repeats: int,
) -> dict:
    evaluator = FloodT2MEvaluator(FLOOD_ROOT, torch.device('cuda'), seed=0)
    runs = []
    for repeat in range(repeats):
        evaluator.seed = 1234 + repeat
        evaluator.prediction_embeddings = [prediction]
        evaluator.ground_truth_embeddings = [ground_truth]
        evaluator.text_embeddings = [text]
        runs.append(evaluator.compute())
    keys = [key for key in runs[0] if key != 'samples']
    return {
        'repeats': repeats,
        'mean': {key: float(np.mean([run[key] for run in runs])) for key in keys},
        'std': {key: float(np.std([run[key] for run in runs], ddof=1)) for key in keys},
        'ci95_half_width': {
            key: float(1.96 * np.std([run[key] for run in runs], ddof=1) / np.sqrt(repeats))
            for key in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--embedding-root', type=Path, default=ROOT / 'data/umt5_xxl_sentence_features')
    parser.add_argument(
        '--flood-protocol',
        type=Path,
        default=ROOT / 'eval/flood_hml_paper_cfg6_20260908',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metric-repeats', type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    dataset = HumanMLMatchedDataset(args.embedding_root, args.flood_protocol)
    if len(dataset) != 4196:
        raise RuntimeError(f'Expected 4196 Flood-matched samples, got {len(dataset)}')
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )

    hml = FLOOD_ROOT / 'raw_data/HumanML3D'
    mean = torch.from_numpy(np.load(hml / 'Mean.npy'))
    std = torch.from_numpy(np.load(hml / 'Std.npy'))
    objective = T2MMeanFlow(OneShotT2MDecoder(), mean, std).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    objective.load_state_dict(checkpoint['model'])
    if 'ema_decoder' in checkpoint:
        objective.decoder.load_state_dict(checkpoint['ema_decoder'])
    objective.eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)

    for sampling_steps in args.steps:
        evaluator = FloodT2MEvaluator(FLOOD_ROOT, device, seed=args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed + sampling_steps * 100000)
        all_keys = []
        all_captions = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f'Flower HML {sampling_steps}-step'):
                condition = batch['condition'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                ground_truth = batch['motion'].to(device, non_blocking=True)
                prediction = objective.sample(condition, mask, steps=sampling_steps, generator=generator)
                prediction_raw = prediction * objective.motion_std + objective.motion_mean
                ground_truth_raw = ground_truth * objective.motion_std + objective.motion_mean
                evaluator.add_batch(
                    prediction_raw,
                    ground_truth_raw,
                    batch['lengths'],
                    batch['tokens'],
                )
                all_keys.append(batch['keys'])
                all_captions.extend(batch['captions'])

        prediction_embeddings = torch.cat(evaluator.prediction_embeddings)
        ground_truth_embeddings = torch.cat(evaluator.ground_truth_embeddings)
        text_embeddings = torch.cat(evaluator.text_embeddings)
        single = evaluator.compute()
        repeated = metric_repeats(
            prediction_embeddings,
            ground_truth_embeddings,
            text_embeddings,
            args.metric_repeats,
        )
        embeddings_path = args.output / f'embeddings_{sampling_steps}step.npz'
        np.savez(
            embeddings_path,
            prediction=prediction_embeddings.numpy(),
            ground_truth=ground_truth_embeddings.numpy(),
            text=text_embeddings.numpy(),
            keys=np.concatenate(all_keys),
            captions=np.asarray(all_captions),
        )
        record = {
            'model': 'FlowerDance T2M tr_t0_full EMA',
            'checkpoint': str(args.checkpoint.resolve()),
            'checkpoint_step': int(checkpoint.get('step', -1)),
            'sampling_steps': sampling_steps,
            'sampling_mode': 'one-shot whole-sequence ODE',
            'split': 'HumanML3D test.txt',
            'samples': len(dataset),
            'protocol': 'exact FloodDiffusion reproduction IDs and captions',
            'single_seed_metrics': single,
            'metric_repeats': repeated,
            'embeddings': str(embeddings_path.resolve()),
        }
        output_path = args.output / f'metrics_{sampling_steps}step.json'
        output_path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
