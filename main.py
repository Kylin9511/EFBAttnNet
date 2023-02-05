import torch
import numpy as np

from utils.configs import args
from utils import logger, Tester
from utils import init_device, init_model
from dataset import MatRawChannelDataLoader
from model import SumRateLoss


def main():
    logger.info("=> PyTorch Version: {}".format(torch.__version__))

    # Environment initialization
    device = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    test_loader = MatRawChannelDataLoader(data_dir=args.test_data_dir, device=device, bs=None)

    # Define model
    model = init_model(args)
    model.to(device)

    # Define loss function
    criterion = SumRateLoss(K=args.users, M=args.antennas_bs).to(device)

    # Inference mode
    if args.evaluate:
        sum_rate = Tester(model, args.power, args.snr_test, device, criterion)(test_loader)
        print(f"\n=! Final test sum_rate: {sum_rate:.3e}\n")
        return


if __name__ == "__main__":
    main()
