from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_flooddiffusion_baseline import copy_released_ema
from t2m_long.evaluator import FloodT2MEvaluator


ROOT = Path('/efs/yangkaixing/FlowerDance_T2M_Long')
FLOOD_ROOT = ROOT / 'third_party/FloodDiffusion'


class HumanMLPaperDataset(Dataset):
    def __init__(self, num_shards: int, shard_index: int, seed: int):
        self.motion_root = FLOOD_ROOT / 'raw_data/HumanML3D/new_joint_vecs'
        self.text_root = FLOOD_ROOT / 'raw_data/HumanML3D/texts'
        ids = [x.strip() for x in (FLOOD_ROOT / 'raw_data/HumanML3D/test.txt').read_text().splitlines() if x.strip()]
        self.items = [(i, name) for i, name in enumerate(ids) if i % num_shards == shard_index]
        self.seed = int(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        global_index, name = self.items[index]
        motion = np.load(self.motion_root / f'{name}.npy').astype(np.float32)
        # This is exactly the released HumanML3D loader's validity rule.
        if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) < 40 or len(motion) > 200 or not np.isfinite(motion).all():
            return None
        annotations = []
        for line in (self.text_root / f'{name}.txt').read_text().splitlines():
            if not line.strip():
                continue
            caption, tokens, _, _ = line.split('#')
            annotations.append((caption, tokens.split()))
        if not annotations:
            return None
        # One deterministic caption per test motion, matching the released
        # loader's one-random-caption convention while making sharding stable.
        caption, tokens = annotations[random.Random(self.seed + global_index * 1000003).randrange(len(annotations))]
        return {
            'motion': torch.from_numpy(motion),
            'length': len(motion),
            'caption': caption,
            'tokens': tokens,
            'key': global_index,
            'name': name,
        }


def collate(items):
    items = [x for x in items if x is not None]
    if not items:
        return None
    return {
        'motion': pad_sequence([x['motion'] for x in items], batch_first=True),
        'lengths': torch.tensor([x['length'] for x in items], dtype=torch.long),
        'captions': [x['caption'] for x in items],
        'tokens': [x['tokens'] for x in items],
        'keys': np.asarray([x['key'] for x in items], dtype=np.int64),
        'names': [x['name'] for x in items],
    }


def load_models(device: torch.device, cfg_scale: float):
    sys.path.insert(0, str(FLOOD_ROOT))
    from models.diffusion_forcing_wan import DiffForcingWanModel
    from models.vae_wan_1d import VAEWanModel

    vae_path = FLOOD_ROOT / 'outputs/vae_1d_z4_step=300000.ckpt'
    model_path = FLOOD_ROOT / 'outputs/20251106_063218_ldf/step_step=50000.ckpt'
    vae = VAEWanModel(input_dim=263, z_dim=4)
    checkpoint = torch.load(vae_path, map_location='cpu', weights_only=False)
    vae.load_state_dict(checkpoint['state_dict'], strict=True)
    if 'ema_state' in checkpoint:
        copy_released_ema(vae, checkpoint['ema_state'])
    model = DiffForcingWanModel(
        checkpoint_path=str(FLOOD_ROOT / 'deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(FLOOD_ROOT / 'deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl'),
        input_dim=4, noise_steps=10, cfg_scale=cfg_scale,
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if 'ema_state' in checkpoint:
        copy_released_ema(model, checkpoint['ema_state'])
    return vae.to(device).eval().requires_grad_(False), model.to(device).eval().requires_grad_(False), model_path, vae_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-scale', type=float, default=6.0)
    parser.add_argument('--denoise-steps', type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    dataset = HumanMLPaperDataset(args.num_shards, args.shard_index, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate)
    vae, model, model_path, vae_path = load_models(device, args.cfg_scale)
    evaluator = FloodT2MEvaluator(FLOOD_ROOT, device, seed=args.seed)
    arrays = {'prediction': [], 'ground_truth': [], 'text': [], 'keys': [], 'captions': []}
    valid = 0
    for batch_index, batch in enumerate(tqdm(loader, desc=f'HML shard {args.shard_index}')):
        if batch is None:
            continue
        lengths = batch['lengths']
        latent_lengths = torch.div(lengths + vae.downsample_factor - 1, vae.downsample_factor, rounding_mode='floor')
        with torch.inference_mode():
            generated = model.generate({'feature_length': latent_lengths, 'text': batch['captions']},
                                       num_denoise_steps=args.denoise_steps)['generated']
            prediction = [vae.decode(x[None].to(device))[0].float() for x in generated]
            prediction_lengths = torch.tensor([len(x) for x in prediction], dtype=torch.long)
            prediction_padded = pad_sequence(prediction, batch_first=True)
            ground_truth = batch['motion'].to(device, non_blocking=True)
            arrays['prediction'].append(evaluator._motion_embeddings(prediction_padded, prediction_lengths).float().cpu().numpy())
            arrays['ground_truth'].append(evaluator._motion_embeddings(ground_truth, lengths).float().cpu().numpy())
            arrays['text'].append(evaluator._text_embeddings(batch['tokens']).float().cpu().numpy())
        arrays['keys'].append(batch['keys'])
        arrays['captions'].extend(batch['captions'])
        valid += len(lengths)
        if (batch_index + 1) % 100 == 0:
            model.text_cache.clear()
    output_path = args.output / f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz'
    np.savez(output_path,
             prediction=np.concatenate(arrays['prediction']),
             ground_truth=np.concatenate(arrays['ground_truth']),
             text=np.concatenate(arrays['text']),
             keys=np.concatenate(arrays['keys']),
             captions=np.asarray(arrays['captions']))
    record = {
        'model': 'FloodDiffusion released HumanML3D checkpoint',
        'checkpoint': str(model_path), 'vae_checkpoint': str(vae_path),
        'cfg_scale': args.cfg_scale, 'denoise_steps': args.denoise_steps,
        'split': 'HumanML3D test.txt', 'seed': args.seed,
        'valid_samples': valid, 'num_shards': args.num_shards, 'shard_index': args.shard_index,
        'embedding_path': str(output_path),
    }
    (args.output / f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
