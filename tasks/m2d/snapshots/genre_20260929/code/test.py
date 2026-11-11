from args import parse_train_opt
from EDGE import EDGE


def test(opt):
    if not opt.checkpoint:
        raise ValueError("Pass an x0 checkpoint with --checkpoint")
    model = EDGE(
        checkpoint_path=opt.checkpoint,
        x0_variant=opt.x0_variant,
        min_t=opt.min_t,
        min_interval=opt.min_interval,
        fm_time_weight_power=opt.fm_time_weight_power,
        cfg_dropout_prob=opt.cfg_dropout_prob,
        cfg_omega=opt.cfg_omega,
        cross_modal_fusion=opt.cross_modal_fusion,
        condition_selection=opt.condition_selection,
        direction_fusion=opt.direction_fusion,
    )
    model.test_loop(opt)

if __name__ == "__main__":
    opt = parse_train_opt()
    test(opt)
