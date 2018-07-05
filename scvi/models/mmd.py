import torch
import torch.nn as nn
import torch.nn.functional as F
#F.cosine_similarity(1,0)
import numpy as np

from torch.distributions import Normal, Uniform
#nn.CosineSimilarity(dim=0, eps=1e-6)
from math import sqrt, pi

def mmd_fourier(x1, x2, bandwidth=2., dim_r=500):
    d = x1.size(1)
    rW_n = sqrt(2. / bandwidth) * Normal(0., 1./sqrt(d)).sample((dim_r,d))
    rb_u = 2 * pi * Uniform(0.,1.).sample((dim_r,))
    rf0 = sqrt(2. / dim_r) * torch.cos(F.linear(x1, rW_n,rb_u))
    rf1 = sqrt(2. / dim_r) * torch.cos(F.linear(x2, rW_n,rb_u))
    result = (torch.pow(rf0.mean(dim=0) - rf1.mean(dim=0),2)).sum()
    return torch.sqrt(result)

def mmd_objective(z, batch_index, n_batch):
    """
    Compute the MMD from latent space and nuisance_id

    Notes:
    Reimplementation in tensorflow of the Variational Fair Autoencoder
    https://arxiv.org/abs/1511.00830
    """

    #mmd_method = mmd_rbf
    mmd_method = mmd_fourier

    z_dim = z.size(1)
    batch_index = batch_index.view(-1)

    # STEP 1: construct lists of samples in their proper batches
    z_part = [z[batch_index==b_i] for b_i in range(n_batch)]

    # STEP 2: add noise to all of them and get the mmd
    mmd = 0
    for j, z_j in enumerate(z_part):
        z0_ = z_j
        aux_z0 = Normal(0., 1.).sample((1,z_dim))
        z0 = torch.cat((z0_, aux_z0), dim=0)
        if len(z_part) == 2:
            z1_ = z_part[j + 1]
            aux_z1 = Normal(0., 1.).sample((1,z_dim))
            z1 = torch.cat((z1_, aux_z1), dim=0)
            return mmd_method(z0, z1)
        z1 = z
        mmd += mmd_method(z0, z1)
    return mmd

