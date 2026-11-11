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
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()

    parts = {key: [] for key in ('prediction', 'ground_truth', 'text', 'keys')}
    for path in sorted(args.input.glob('shard_*.npz')):
        with np.load(path) as shard:
            for key in parts:
                parts[key].append(shard[key].copy())
    values = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(values['keys'], kind='stable')
    values = {key: value[order] for key, value in values.items()}

    evaluator = FloodT2MEvaluator(FLOOD_ROOT, torch.device('cuda'), seed=0)
    prediction = torch.from_numpy(values['prediction'])
    ground_truth = torch.from_numpy(values['ground_truth'])
    text = torch.from_numpy(values['text'])
    runs = []
    for seed in range(args.repeats):
        evaluator.seed = 1234 + seed
        evaluator.prediction_embeddings = [prediction]
        evaluator.ground_truth_embeddings = [ground_truth]
        evaluator.text_embeddings = [text]
        runs.append(evaluator.compute())

    keys = [key for key in runs[0] if key != 'samples']
    summary = {
        'samples': len(values['keys']),
        'repeats': args.repeats,
        'mean': {key: float(np.mean([run[key] for run in runs])) for key in keys},
        'std': {key: float(np.std([run[key] for run in runs], ddof=1)) for key in keys},
        'ci95_half_width': {
            key: float(1.96 * np.std([run[key] for run in runs], ddof=1) / np.sqrt(args.repeats))
            for key in keys
        },
        'runs': runs,
    }
    output = args.input / f'metrics_10step_cfg6_{args.repeats}metric_repeats.json'
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ('samples', 'repeats', 'mean', 'std', 'ci95_half_width')}, indent=2))


if __name__ == '__main__':
    main()
