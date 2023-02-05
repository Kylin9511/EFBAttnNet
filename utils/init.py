import os
import random
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from model import SubArrayEFBRefineNet, SubArrayEFBAttnNet
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f"taskset -p {affinity} {os.getpid()}")

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU")

    return device


def init_model(args, verbose=True):
    model_params = dict(
        M=args.antennas_bs,
        L=args.pilots,
        P=args.power,
        K=args.users,
        B=args.feedback_bits,
        anneal_init=args.anneal_init,
        anneal_rate=args.anneal_rate,
    )
    if args.model == "SubArrayEFBRefineNet":
        model = SubArrayEFBRefineNet(**model_params)
    elif args.model == "SubArrayEFBAttnNet":
        model = SubArrayEFBAttnNet(**model_params)
    else:
        raise ValueError(f"Illegal model name {args.model}")

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained, map_location=torch.device("cpu"))["state_dict"]
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    if verbose is True:
        # Model flops and params counting
        model.eval()
        fake_H_real = torch.rand((1, args.antennas_bs, args.users))
        fake_H_imag = torch.rand((1, args.antennas_bs, args.users))
        fake_noise_std = torch.tensor((1.0,))
        inputs = (fake_H_real, fake_H_imag, fake_noise_std)
        print(flop_count_table(FlopCountAnalysis(model, inputs)))

        # Model info logging
        logger.info(f"=> Model Name: EFBNet [pretrained: {args.pretrained}]")
        logger.info(f"=> system args: {args}")
        logger.info(f"\n{line_seg}\n{model}\n{line_seg}\n")

    return model
