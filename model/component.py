import torch
from torch import nn
import numpy as np

__all__ = ["merge_hybrid_beamformer", "SubArrayHybridBeamformingNet", "SubArrayPilotNet", "SigmoidT"]


def merge_hybrid_beamformer(A_real, A_imag, D_real, D_imag, C=None):
    r"""Merge analog precoder and digital precoder into a final precoder.

    Args:
        A_real: the real part of the analog precoder
        A_imag: the imaginary part of the analog precoder
        D_real: the real part of the digital precoder
        D_imag: the imaginary part of the digital precoder
        C: the dynamic/fixed antenna selection matrix if given
    """

    if C is not None:
        # A: N*M*M, C: N*M*K, D: N*K*K
        ArCDr = torch.matmul(torch.matmul(A_real, C), D_real)  # N*M*K
        ArCDi = torch.matmul(torch.matmul(A_real, C), D_imag)  # N*M*K
        AiCDr = torch.matmul(torch.matmul(A_imag, C), D_real)  # N*M*K
        AiCDi = torch.matmul(torch.matmul(A_imag, C), D_imag)  # N*M*K
        V_real = ArCDr - AiCDi  # N*(M*K)
        V_imag = ArCDi + AiCDr  # N*(M*K)
    else:
        # A: N*M*K, D: N*K*K
        ArDr = torch.matmul(A_real, D_real)  # N*M*K
        ArDi = torch.matmul(A_real, D_imag)  # N*M*K
        AiDr = torch.matmul(A_imag, D_real)  # N*M*K
        AiDi = torch.matmul(A_imag, D_imag)  # N*M*K
        V_real = ArDr - AiDi
        V_imag = ArDi + AiDr
    return V_real, V_imag


class SubArrayHybridBeamformingNet(nn.Module):
    def __init__(self, M, K, P, feature_dim=512):
        super(SubArrayHybridBeamformingNet, self).__init__()
        assert M % K == 0, f"Currently we assume the number of antennas {M} can be equally divided into {K} groups"
        self.M = M
        self.K = K
        self.P = P
        antenna_selection_mask = torch.block_diag(*[torch.ones((M // K, 1))] * K)
        self.register_buffer("C", antenna_selection_mask.clone().detach())

        self.FC_A_theta = nn.Linear(feature_dim, M)
        self.FC_D_real = nn.Linear(feature_dim, K * K)
        self.FC_D_imag = nn.Linear(feature_dim, K * K)

    def forward(self, x):
        batch_size = x.size(0)  # N

        # Produce the low resolution Analog precoding matrix with q-bit phase shifter
        A_theta = self.FC_A_theta(x)  # N*M
        A_real = torch.diag_embed(torch.cos(A_theta))  # N*M -> N*M*M
        A_imag = torch.diag_embed(torch.sin(A_theta))  # N*M -> N*M*M

        # Produce the Digital precoding matrix
        D_real = self.FC_D_real(x).view(batch_size, self.K, self.K)  # N*K*K
        D_imag = self.FC_D_imag(x).view(batch_size, self.K, self.K)  # N*K*K

        # Normalization of digital precoding matrix
        V_real, V_imag = merge_hybrid_beamformer(A_real, A_imag, D_real, D_imag, self.C)
        V_F_norm = torch.sqrt(torch.sum(V_real**2 + V_imag**2, dim=(1, 2))).view(batch_size, 1, 1)  # N*1*1
        D_real = np.sqrt(self.P) * torch.div(D_real, V_F_norm)
        D_imag = np.sqrt(self.P) * torch.div(D_imag, V_F_norm)
        return A_real, A_imag, D_real, D_imag, self.C


class SubArrayPilotNet(nn.Module):
    def __init__(self, M, L, P):
        super(SubArrayPilotNet, self).__init__()
        self.P = P
        self.M = M

    def forward(self, X_theta, H_real, H_imag, noise_std):
        X_real = np.sqrt(self.P / self.M) * torch.cos(X_theta)
        X_imag = np.sqrt(self.P / self.M) * torch.sin(X_theta)
        Y_real = torch.einsum("LM,NM -> NL", [X_real, H_real]) - torch.einsum("LM,NM -> NL", [X_imag, H_imag])
        Y_imag = torch.einsum("LM,NM -> NL", [X_real, H_imag]) + torch.einsum("LM,NM -> NL", [X_imag, H_real])
        Y = torch.concat([Y_real, Y_imag], dim=1)
        Y_N = Y + torch.normal(mean=torch.zeros_like(Y), std=noise_std)
        return Y_N


class SigmoidT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, anneal_factor):
        y_tanh = torch.sigmoid(anneal_factor * y) * 2 - 1
        ctx.save_for_backward(y_tanh, torch.tensor(anneal_factor))
        y_sign = torch.sign(y)
        return y_sign

    @staticmethod
    def backward(ctx, grad_output):
        y_tanh, anneal_factor = ctx.saved_tensors
        grad = anneal_factor.item() * (y_tanh + 1) * (1 - y_tanh) / 2  # sigmoid gradient
        grad_input = grad_output * grad
        return grad_input, None
