import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument("--processed_data_dir", type=str, default="data/dataset_backups/", help="Dataset backup path",)
    parser.add_argument("--render_dir", type=str, default="renders/", help="Sample render path")
    parser.add_argument("--eval_dir", type=str, default="eval/", help="Sample render path")

    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument("--wandb_pj_name", type=str, default="EDGE", help="project name")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--force_reload", action="store_true", help="force reloads the datasets")
    parser.add_argument("--no_cache", action="store_true", help="don't reuse / cache loaded dataset")
    parser.add_argument("--save_interval", type=int, default=50, help='Log model after every "save_period" epoch',)
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument( "--checkpoint", type=str, default="", help="trained checkpoint path (optional)")
    parser.add_argument(
        "--x0_variant",
        choices=(
            "x0_flow",
            "diffusion",
            "t0_only",
            "tr_only",
            "tr_t0",
            "tr_t0_aux",
            "tr_t0_aux_rec_only",
            "tr_t0_aux_cfg_baked",
            "tr_t0_full",
            "tr_t0_full_avg",
            "tr_cfg_baked",
        ),
        default="tr_only",
        help=(
            "x0 MeanFlow ablation. tr_t0 is a backward-compatible alias for "
            "tr_t0_aux. tr_t0_full_avg averages both complete branches. "
            "tr_t0_aux_rec_only and tr_t0_aux_cfg_baked are strict ablations "
            "of the aux topology. CFG variants use training-time CFG but one conditional "
            "network call at inference."
        ),
    )
    parser.add_argument("--fm_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=0.636)
    parser.add_argument("--vel_weight", type=float, default=0.323)
    parser.add_argument("--acc_weight", type=float, default=0.2)
    parser.add_argument("--fk_weight", type=float, default=0.646)
    parser.add_argument("--floor_weight", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--min_t", type=float, default=0.05)
    parser.add_argument("--min_interval", type=float, default=0.05)
    parser.add_argument("--fm_time_weight_power", type=float, default=2.0)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)
    parser.add_argument("--cfg_omega", type=float, default=2.0)
    parser.add_argument(
        "--cross_modal_fusion",
        choices=("add", "adain"),
        default="add",
        help="Cross-modal injection used in every dance block.",
    )
    parser.add_argument(
        "--condition_selection",
        action="store_true",
        help="Inject the frame-wise condition directly into Mamba delta/B/C selection parameters.",
    )
    parser.add_argument(
        "--direction_fusion",
        choices=("sum", "cond_gate"),
        default="sum",
        help="Fuse forward/backward Mamba streams by summation or a condition-driven gate.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=10,
        help="Number of few-step x0 updates used for training-time evaluation",
    )
    parser.add_argument(
        "--eval_sampling_steps",
        type=int,
        nargs="+",
        default=(1, 2, 4),
        help="Few-step checkpoints evaluated whenever --save_interval is reached.",
    )
    parser.add_argument(
        "--skip_metric",
        action="store_true",
        help="During test.py, save generated motions without running metrics.",
    )
    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument("--out_length", type=float, default=40, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    opt = parser.parse_args()
    return opt
