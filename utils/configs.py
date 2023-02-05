import argparse

parser = argparse.ArgumentParser(description="Feedback beamforming PyTorch Training")


# ========================== System arguments ==========================

parser.add_argument("-M", "--antennas-bs", required=True, type=int, help="the number of antennas at the base station")
parser.add_argument("-K", "--users", required=True, type=int, help="the number of users")
parser.add_argument("-L", "--pilots", required=True, type=int, help="the number of pilots")
parser.add_argument("-Lp", "--channel-paths", required=True, type=int, help="the number of clustered channel paths")
parser.add_argument("-B", "--feedback-bits", required=True, type=int, help="the number of feedback bits")
parser.add_argument("-P", "--power", default=1, type=int, help="the power constraint of the BS transmitted symbol")
parser.add_argument("-SNRTest", "--snr-test", default=10, type=int, help="the signal to noise ratio for inference")

# ========================== Inference arguments ==========================
parser.add_argument("--model", default="EFBNet", type=str, help="the name of the model")
parser.add_argument("--anneal-init", default=1, type=float, help="the init annealing param for sigmoid-adjusted ST")
parser.add_argument("--anneal-rate", default=1.001, type=float, help="the annealing rate for sigmoid-adjusted ST")

parser.add_argument("--test-data-dir", type=str, required=True, help="the path of test dataset.")
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--cpu", action="store_true", help="disable GPU training (default: False)")
parser.add_argument("--cpu-affinity", default=None, type=str, help='CPU affinity, like "0xffff"')

parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument(
    "--pretrained",
    type=str,
    default=None,
    help="using locally pre-trained model. The path of pre-trained model should be given",
)
parser.add_argument(
    "--resume",
    type=str,
    metavar="PATH",
    default=None,
    help="path to latest checkpoint (default: none)",
)

args = parser.parse_args()


if __name__ == "__main__":
    print(f"The members: {args}")
