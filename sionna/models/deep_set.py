from typing import Type

import torch
from torch import nn


class EquivariantLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.Gamma = nn.Linear(in_channels, out_channels, bias=False)
        self.Lambda = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [batch_size, n_elements, in_channels]
        # return: [batch_size, n_elements, out_channels]
        xm, _ = torch.max(x, dim=1, keepdim=True)
        return self.Lambda(x) - self.Gamma(xm)


class EquivariantDeepSet(nn.Module):
    def __init__(self, in_channels: int, net_arch: list[int], act_fn: Type[nn.Module]) -> None:
        super().__init__()
        self.net = nn.Sequential()
        net_arch = [in_channels] + net_arch
        for i in range(len(net_arch) - 2):
            self.net.append(EquivariantLayer(net_arch[i], net_arch[i + 1]))
            self.net.append(act_fn())
        self.net.append(EquivariantLayer(net_arch[-2], net_arch[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, n_elements, in_channels]
        # return: [batch_size, n_elements, net_arch[-1]]
        return torch.squeeze(self.net(x), dim=-1)


class InvariantDeepSet(nn.Module):
    def __init__(self, in_channels: int, psi_arch: list[int], rho_arch: list[int], act_fn: Type[nn.Module]) -> None:
        super().__init__()
        self.psi = EquivariantDeepSet(in_channels, psi_arch, act_fn)
        rho_arch = [psi_arch[-1]] + rho_arch
        self.rho = nn.Sequential()
        for i in range(len(rho_arch) - 1):
            self.rho.append(nn.Linear(rho_arch[i], rho_arch[i + 1]))
            self.rho.append(act_fn())
        self.rho.append(nn.Linear(rho_arch[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        x = torch.mean(self.psi(x), dim=1)
        return torch.squeeze(self.rho(x), dim=-1)
