import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as sci
from einops import rearrange
from typing import Union
from collections import OrderedDict

from .component import SubArrayPilotNet, SigmoidT, SubArrayHybridBeamformingNet, merge_hybrid_beamformer
from utils import logger

__all__ = ["SubArrayEFBAttnNet"]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            modules = [
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]
            self.layers.append(nn.ModuleList(modules))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y


class UserNet(nn.Module):
    def __init__(self, M, L, P, B, expand=8):
        super(UserNet, self).__init__()
        self.pilot = SubArrayPilotNet(M, L, P)
        module_list = [
            ("conv", nn.Conv1d(2, expand, 9, stride=1, padding=4, bias=False)),
            ("bn", nn.BatchNorm1d(expand)),
            ("relu", nn.ReLU()),
        ]
        self.conv_net = nn.Sequential(OrderedDict(module_list))
        self.transformer = Transformer(dim=L, depth=3, heads=4, dim_head=L, mlp_dim=2 * L)
        module_list = [
            ("fc1", nn.Linear(expand * L, 2 * B)),
            ("bn1", nn.BatchNorm1d(2 * B)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(2 * B, B)),
            ("bn2", nn.BatchNorm1d(B)),
            ("sigmoid", nn.Sigmoid()),
        ]
        self.fc_net = nn.Sequential(OrderedDict(module_list))

    def forward(self, X_theta, H_real, H_imag, anneal_factor, noise_std):
        y = self.pilot(X_theta, H_real, H_imag, noise_std)

        batch_size = H_real.size(0)
        y = y.view(batch_size, 2, -1)
        y_expand = self.conv_net(y)
        y = self.transformer(y_expand).view(batch_size, -1)
        y = self.fc_net(y)
        y = y - 0.5  # from (0,1) to (-0.5,0.5)
        q = SigmoidT().apply(y, anneal_factor)
        return q


class DecoderNet(nn.Module):
    def __init__(self, M, K, P, B):
        super(DecoderNet, self).__init__()
        self.K = K
        self.B = B
        expand = 512 // B

        module_list = [
            ("conv", nn.Conv1d(K, expand * K, 9, stride=1, padding=4, bias=False)),
            ("bn", nn.BatchNorm1d(expand * K)),
            ("se", SEBlock(expand * K, reduction=K)),
            ("relu", nn.ReLU()),
        ]
        self.se_expander = nn.Sequential(OrderedDict(module_list))
        module_list = [
            ("fc1", nn.Linear(expand * K * B, 512)),
            ("bn1", nn.BatchNorm1d(512)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(512, 512)),
            ("bn2", nn.BatchNorm1d(512)),
            ("relu2", nn.ReLU()),
        ]
        self.fc_net = nn.Sequential(OrderedDict(module_list))
        self.beamformer = SubArrayHybridBeamformingNet(M, K, P, feature_dim=512)

    def forward(self, q):
        batch_size = q.size(0)
        q = q.view(batch_size, self.K, self.B)
        feature = self.se_expander(q).view(batch_size, -1)
        feature = self.fc_net(feature)
        return self.beamformer(feature)


class SubArrayEFBAttnNet(nn.Module):
    def __init__(self, M: int, L: int, P: int, K: int, B: int, anneal_init: float, anneal_rate: float):
        r"""
        Args:
            M: number of antennas at the base station
            L: number of channel paths
            P: the power constraint of the base station
            K: number of users
            B: number of available feedback bits
            anneal_init: initial anneal factor of sigmoidT function
            anneal_rate: anneal factor update rate of sigmoidT function
        """

        super(SubArrayEFBAttnNet, self).__init__()
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
        logger.info(f"=> Model: Using SubArrayEFBAttnNet with M={M}, L={L}, P={P}, K={K}, B={B}")
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
