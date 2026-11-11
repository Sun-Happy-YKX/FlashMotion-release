from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        x0_variant=opt.x0_variant,
        loss_weights={
            "fm": opt.fm_weight,
            "rec": opt.rec_weight,
            "vel": opt.vel_weight,
            "acc": opt.acc_weight,
            "fk": opt.fk_weight,
            "floor": opt.floor_weight,
        },
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        min_t=opt.min_t,
        min_interval=opt.min_interval,
        fm_time_weight_power=opt.fm_time_weight_power,
        cfg_dropout_prob=opt.cfg_dropout_prob,
        cfg_omega=opt.cfg_omega,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
