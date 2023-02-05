import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as sci
from typing import Union
from collections import OrderedDict

from .component import SubArrayPilotNet, SigmoidT, SubArrayHybridBeamformingNet, merge_hybrid_beamformer
from utils import logger

__all__ = ["SubArrayEFBRefineNet"]


class UserNet(nn.Module):
    def __init__(self, M, L, P, B):
        super(UserNet, self).__init__()
        self.pilot = SubArrayPilotNet(M, L, P)
        module_list = [
            ("conv1", nn.Conv1d(2, 8, 5, stride=1, padding=2, bias=False)),
            ("bn1", nn.BatchNorm1d(8)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv1d(8, 16, 5, stride=1, padding=2, bias=False)),
            ("bn2", nn.BatchNorm1d(16)),
            ("relu2", nn.ReLU()),
            ("conv3", nn.Conv1d(16, 2, 5, stride=1, padding=2, bias=False)),
            ("bn3", nn.BatchNorm1d(2)),
        ]
        self.refine_encoder = nn.Sequential(OrderedDict(module_list))
        module_list = [
            ("fc_1024", nn.Linear(2 * L, 1024)),
            ("bn_1", nn.BatchNorm1d(1024)),
            ("relu_1", nn.ReLU()),
            ("fc_512", nn.Linear(1024, 512)),
            ("bn_2", nn.BatchNorm1d(512)),
            ("relu_2", nn.ReLU()),
            ("fc_B", nn.Linear(512, B)),
            ("bn_3", nn.BatchNorm1d(B)),
            ("sigmoid", nn.Sigmoid()),
        ]
        self.fc_encoder = nn.Sequential(OrderedDict(module_list))

    def forward(self, X_theta, H_real, H_imag, anneal_factor, noise_std):
        y = self.pilot(X_theta, H_real, H_imag, noise_std)

        batch_size = H_real.size(0)
        y = y.view(batch_size, 2, -1)
        residual = self.refine_encoder(y)
        y = (y + residual).view(batch_size, -1)

        y = self.fc_encoder(y)
        y = y - 0.5  # from (0,1) to (-0.5,0.5)
        q = SigmoidT().apply(y, anneal_factor)
        return q


class DecoderNet(nn.Module):
    def __init__(self, M, K, P, B):
        super(DecoderNet, self).__init__()
        module_list = [
            ("fc1", nn.Linear(K * B, 2048)),
            ("bn1", nn.BatchNorm1d(2048)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(2048, 1024)),
            ("bn2", nn.BatchNorm1d(1024)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(1024, 1024)),
            ("bn3", nn.BatchNorm1d(1024)),
            ("relu3", nn.ReLU()),
        ]
        self.decoder = nn.Sequential(OrderedDict(module_list))
        self.beamformer = SubArrayHybridBeamformingNet(M, K, P, feature_dim=1024)

    def forward(self, q):
        feature = self.decoder(q)
        return self.beamformer(feature)


class SubArrayEFBRefineNet(nn.Module):
    def __init__(self, M: int, L: int, P: int, K: int, B: int, anneal_init: float, anneal_rate: float):
        r"""
        Args:
            M: number of antennas at the base station
            L: number of channel paths
            P: the power constraint of the base station
            K: number of users
            B: number of available feedback bits
            q: number of phase shifter quantization bits, None for no quantization
            anneal_init: initial anneal factor of sigmoidT function
            anneal_rate: anneal factor update rate of sigmoidT function
        """

        super(SubArrayEFBRefineNet, self).__init__()
        self.users = nn.ModuleList()
        for _ in range(K):
            self.users.append(UserNet(M, L, P, B))
        self.base_station = DecoderNet(M, K, P, B)
        self.anneal_factor = anneal_init
        self.anneal_rate = anneal_rate

        # take L pilots from DFT matrix (L * M) and register it as a global Parameter
        DFT_Matrix = sci.dft(M)
        X = np.angle(DFT_Matrix[:: int(np.floor(M / L)), :][:L, :])  # take angle for the hybrid beamforming arch
        self.X_theta = nn.Parameter(torch.tensor(X, dtype=torch.float32))

        assert isinstance(self.anneal_rate, float) and self.anneal_rate > 1, self.anneal_rate
        logger.info(f"=> Model: Using SubArrayEFBRefineNet with M={M}, L={L}, P={P}, K={K}, B={B}")
        logger.info(f"   The SigmoidT is activated with anneal_init={anneal_init} and anneal_rate={anneal_rate}")

    def forward(self, H_real, H_imag, noise_std):
        feedback_bits = []
        for idx, user_model in enumerate(self.users):
            q = user_model(
                X_theta=self.X_theta,
                H_real=H_real[:, :, idx],
                H_imag=H_imag[:, :, idx],
                anneal_factor=self.anneal_factor,
                noise_std=noise_std,
            )
            feedback_bits.append(q)
        feedback_bits = torch.cat(feedback_bits, dim=1)
        self.anneal_factor = min(self.anneal_factor * self.anneal_rate, 10)

        # Design the targeted sub-connected hybrid beamformer
        A_real, A_imag, D_real, D_imag, C = self.base_station(feedback_bits)

        # Merge the hybrid beamformer for easier sum rate calculation
        V_real, V_imag = merge_hybrid_beamformer(A_real, A_imag, D_real, D_imag, C)
        return V_real, V_imag
