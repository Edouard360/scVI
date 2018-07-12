from math import sqrt, pi

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Uniform


def mmd_fourier(x1, x2, bandwidth=2., dim_r=500):
    d = x1.size(1)
    rw_n = sqrt(2. / bandwidth) * Normal(0., 1. / sqrt(d)).sample((dim_r, d)).type(x1.type())
    rb_u = 2 * pi * Uniform(0., 1.).sample((dim_r,)).type(x1.type())
    rf0 = sqrt(2. / dim_r) * torch.cos(F.linear(x1, rw_n, rb_u))
    rf1 = sqrt(2. / dim_r) * torch.cos(F.linear(x2, rw_n, rb_u))
    result = (torch.pow(rf0.mean(dim=0) - rf1.mean(dim=0), 2)).sum()
    return torch.sqrt(result)


def mmd_objective(z, batch_index, n_batch):
    """
    Compute the MMD from latent space and nuisance_id

    Notes:
    For the reimplementation in pytorch of the Variational Fair Autoencoder
    https://arxiv.org/abs/1511.00830
    """

    mmd_method = mmd_fourier

    z_dim = z.size(1)
    batch_index = batch_index.view(-1)

    # STEP 1: construct lists of samples in their proper batches
    z_part = [z[batch_index == b_i] for b_i in range(n_batch)]

    # STEP 2: add noise to all of them and get the mmd
    mmd = 0
    for j, z_j in enumerate(z_part):
        z0_ = z_j
        aux_z0 = Normal(0., 1.).sample((1, z_dim)).type(z0_.type())
        z0 = torch.cat((z0_, aux_z0), dim=0)
        if len(z_part) == 2:
            z1_ = z_part[j + 1]
            aux_z1 = Normal(0., 1.).sample((1, z_dim)).type(z1_.type())
            z1 = torch.cat((z1_, aux_z1), dim=0)
            return mmd_method(z0, z1)
        z1 = z
        mmd += mmd_method(z0, z1)
    return mmd
