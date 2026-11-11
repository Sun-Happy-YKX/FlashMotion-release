import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.evaluator import FloodT2MEvaluator

ROOT = Path('/efs/yangkaixing/FlowerDance_T2M_Long')
FLOOD_ROOT = ROOT / 'third_party/FloodDiffusion'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    paths = sorted(args.input.glob('shard_*.npz'))
    if not paths:
        raise RuntimeError('No HML shards found')
    values = {key: [] for key in ('prediction', 'ground_truth', 'text', 'keys', 'captions')}
    for path in paths:
        with np.load(path) as shard:
            for key in values:
                values[key].append(shard[key].copy())
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    order = np.argsort(values['keys'], kind='stable')
    values = {key: value[order] for key, value in values.items()}
    evaluator = FloodT2MEvaluator(FLOOD_ROOT, torch.device('cuda'), seed=args.seed)
    evaluator.prediction_embeddings = [torch.from_numpy(values['prediction'])]
    evaluator.ground_truth_embeddings = [torch.from_numpy(values['ground_truth'])]
    evaluator.text_embeddings = [torch.from_numpy(values['text'])]
    metrics = evaluator.compute()
    record = {
        'model': 'FloodDiffusion released HumanML3D checkpoint',
        'checkpoint': str(FLOOD_ROOT / 'outputs/20251106_063218_ldf/step_step=50000.ckpt'),
        'vae_checkpoint': str(FLOOD_ROOT / 'outputs/vae_1d_z4_step=300000.ckpt'),
        'cfg_scale': 6.0, 'denoise_steps': 10,
        'split': 'HumanML3D test.txt', 'samples': len(values['keys']),
        'metrics': metrics,
    }
    output = args.input / 'metrics_10step_cfg6.json'
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
