import torch
import torch.nn as nn
from torch.distributions import Normal

from scvi.models.flow import Coupling, Permutation, ActNorm


# GLOW model
class GLOW(nn.Module):
    r"""Variational auto-encoder model."""

    def __init__(self, n_input, n_layers=1, log_variational=True):
        super(GLOW, self).__init__()
        self.log_variational = log_variational
        self.n_layers = n_layers
        self.act_norms = nn.ModuleList([ActNorm(n_input) for _ in range(n_layers)])
        self.permutations = nn.ModuleList([Permutation(n_input) for _ in range(n_layers)])
        self.couplings = nn.ModuleList([Coupling(n_input) for _ in range(n_layers)])

    def initialize(self, x):
        z = x
        log_det = torch.zeros(x.size(0)).to(x.device)
        if self.log_variational:
            z = torch.log(1 + z)
            log_det += (- z.sum(dim=1))

        mean = torch.mean(z, dim=0)
        log_var = 0.5 * torch.log(torch.mean((z - mean) ** 2, dim=0) + 1e-6)

        self.act_norms[0].reset_parameters(mean, log_var)

        for act_norm, permutation, coupling in zip(self.act_norms[1:], self.permutations[:-1], self.couplings[:-1]):
            z = Normal(torch.zeros_like(z), torch.ones_like(z)).sample()
            z, log_det = permutation(z, log_det)
            z, log_det = coupling(z, log_det)
            mean = torch.mean(z, dim=0)
            log_var = 0.5 * torch.log(torch.mean((z - mean) ** 2, dim=0) + 1e-6)
            act_norm.reset_parameters(mean, log_var)

    def forward(self, x):
        z = x
        log_det = torch.zeros(x.size(0)).to(x.device)
        if self.log_variational:
            z = torch.log(1 + z)
            log_det += (- z.sum(dim=1))

        for i, (permutation, coupling, act_norm) in enumerate(zip(self.permutations, self.couplings, self.act_norms)):
            z, log_det = act_norm(z, log_det)
            z, log_det = permutation(z, log_det)
            z, log_det = coupling(z, log_det)
        return z, log_det

    def loss(self, x):
        z, log_det = self(x)
        log_p_z = (Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z)).sum(dim=1)
        return - (log_p_z + log_det)
