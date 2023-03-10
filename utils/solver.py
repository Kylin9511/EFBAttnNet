import time
import os
import torch
import numpy as np

from utils import logger
from utils.statistics import AverageMeter

__all__ = ["Tester"]


class Tester:
    r"""The testing interface for classification"""

    def __init__(self, model, power, snr_test, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        noise_std = np.sqrt(1 / 2) * np.sqrt(power / 10 ** (snr_test / 10))
        self.noise_std = torch.tensor(noise_std, dtype=torch.float32).to(self.device)

    def __call__(self, test_loader, verbose=False):
        r"""Runs the testing procedure.

        Args:
            test_loader (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            sum_rate = self._iteration(test_loader)
        if verbose:
            print(f"\n=> Test result: \nsum_rate: {sum_rate:.3e}")
        return sum_rate

    def _iteration(self, data_loader):
        r"""protected function which test the model on given data loader for one epoch."""

        iter_rate = AverageMeter("Iter sum_rate")
        iter_time = AverageMeter("Iter time")
        time_tmp = time.time()

        for batch_idx, (H_real, H_imag) in enumerate(data_loader):
            H_real = H_real.to(self.device)
            H_imag = H_imag.to(self.device)

            V_r, V_i = self.model(H_real, H_imag, self.noise_std)
            loss = self.criterion(H_real, H_imag, V_r, V_i, 2 * self.noise_std**2)
            sum_rate = -loss

            # Log and update
            iter_rate.update(sum_rate)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f"[{batch_idx + 1}/{len(data_loader)}] sum_rate: {iter_rate.avg:.3f}")

        logger.info(f"=> Test sum_rate: {iter_rate.avg:.3e}\n")

        return iter_rate.avg
