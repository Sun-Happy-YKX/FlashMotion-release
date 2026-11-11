from args import parse_train_opt
from EDGE import EDGE


def inpaint(opt):
    if not opt.checkpoint:
        raise ValueError("Pass an x0 checkpoint with --checkpoint")
    model = EDGE(checkpoint_path=opt.checkpoint, x0_variant=opt.x0_variant)
    model.inpaint_loop(opt)

if __name__ == "__main__":
    opt = parse_train_opt()
    inpaint(opt)
