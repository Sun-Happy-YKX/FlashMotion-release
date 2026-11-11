from pathlib import Path

import numpy as np

from metrics.tools.utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
)
from t2m_long.evaluator import stable_frechet_distance


folder = Path('/efs/yangkaixing/FlowerDance_T2M_Long/eval/flood_hml_paper_cfg6_20260908')
predictions = []
ground_truth = []
for path in sorted(folder.glob('shard_*.npz')):
    with np.load(path) as shard:
        predictions.append(shard['prediction'].copy())
        ground_truth.append(shard['ground_truth'].copy())
predictions = np.concatenate(predictions)
ground_truth = np.concatenate(ground_truth)
pred_mean, pred_cov = calculate_activation_statistics_np(predictions)
gt_mean, gt_cov = calculate_activation_statistics_np(ground_truth)
print('official_scipy_fid', calculate_frechet_distance_np(gt_mean, gt_cov, pred_mean, pred_cov))
print('stable_torch_fid', stable_frechet_distance(gt_mean, gt_cov, pred_mean, pred_cov))
