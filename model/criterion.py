import torch
from torch import nn

__all__ = ["SumRateLoss"]


class SumRateLoss(nn.Module):
    def __init__(self, K, M) -> None:
        super(SumRateLoss, self).__init__()
        self.K = K
        self.M = M

    def forward(self, H_real, H_imag, V_real, V_imag, noise_power):
        batch_size = H_real.size(0)
        rate = torch.zeros(batch_size, self.K).to(H_real.device)
        for k1 in range(self.K):
            for k2 in range(self.K):
                HrVr = torch.einsum("NM,NM -> N", [H_real[:, :, k1], V_real[:, :, k2]])
                HiVi = torch.einsum("NM,NM -> N", [H_imag[:, :, k1], V_imag[:, :, k2]])
                HrVi = torch.einsum("NM,NM -> N", [H_real[:, :, k1], V_imag[:, :, k2]])
                HiVr = torch.einsum("NM,NM -> N", [H_imag[:, :, k1], V_real[:, :, k2]])
                real_part = HrVr - HiVi
                imag_part = HrVi + HiVr
                norm2_hv = real_part**2 + imag_part**2
                if k1 == k2:
                    nom = norm2_hv
                if k2 == 0:
                    nom_denom = norm2_hv + noise_power
                else:
                    nom_denom = nom_denom + norm2_hv
            denom = nom_denom - nom
            rate[:, k1] = torch.log2(1 + torch.div(nom, denom))
        loss = -torch.mean(torch.sum(rate, dim=1))
        return loss
