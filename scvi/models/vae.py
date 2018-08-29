# -*- coding: utf-8 -*-
"""Main module."""

import torch
from torch.distributions import kl_divergence as kl

torch.backends.cudnn.benchmark = True

from torch import nn as nn
from torch.distributions import Normal

import torch
import torch.nn.functional as F
import numpy as np

def compute_log_likelihood(vae, posterior):
    log_lkl = 0
    for i_batch, tensors in enumerate(posterior):
        sample_batch, local_l_mean, local_l_var, _, _ = tensors[:5]  # general fish case
        reconst_loss, kl_divergence = vae(sample_batch, local_l_mean, local_l_var)
        log_lkl += torch.sum(reconst_loss).item()
    n_samples = len(posterior.indices)
    return log_lkl / n_samples


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    case_zero = (F.softplus((- pi + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps)))
                 - F.softplus(-pi))

    case_non_zero = - pi - F.softplus(-pi) + theta * torch.log(theta + eps) - theta * torch.log(
        theta + mu + eps) + x * torch.log(mu + eps) - x * torch.log(theta + mu + eps) + torch.lgamma(
        x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)

    res = torch.mul((x < eps).type(torch.float32), case_zero) + torch.mul((x > eps).type(torch.float32), case_non_zero)
    return torch.sum(res, dim=-1)


class BatchNorm(nn.BatchNorm1d):
    def reset_parameters(self):
        print("Correct batch norm instance")
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

class Linear(nn.Linear):
    def reset_parameters(self):
        self.weight.data.normal_(0.0, 0.01).clamp_(min=-0.02, max=0.02)
        if self.bias is not None:
            self.bias.data.zero_()


class FCLayers(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout_rate=0.1):
        super(FCLayers, self).__init__()
        if dropout_rate>0:
            self.fc_layers = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                Linear(n_in, n_out),
                BatchNorm(n_out, momentum=.01, eps=0.001),
                nn.ReLU()
            )
        else:
            self.fc_layers = nn.Sequential(
                Linear(n_in, n_out),
                BatchNorm(n_out, momentum=.01, eps=0.001),
                nn.ReLU()
            )

    def forward(self, x):
        return self.fc_layers(x)


# Encoder
class Encoder(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128):
        super(Encoder, self).__init__()
        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden)
        self.mean_encoder = Linear(n_hidden, n_output)
        self.var_encoder = Linear(n_hidden, n_output)

    def reparameterize(self, mu, var):
        # b,n = mu.size()
        # np.random.normal(0, 1, size=(b,n))

        #answer = mu + var.sqrt() * torch.from_numpy(next(self.iter_samples)).to(device=var.device).type(var.dtype)

        return Normal(mu, var.sqrt()).rsample()#answer

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = self.reparameterize(q_m, q_v)
        return q_m, q_v, latent


class DecoderSCVI(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super(DecoderSCVI, self).__init__()
        n_hidden = 128
        self.px_decoder = FCLayers(n_in=n_input, n_out=n_hidden, dropout_rate=0)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        #self.px_r_decoder = Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor):
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        return px_scale, px_rate, px_dropout


# VAE model
class VAE(nn.Module):
    def __init__(self, n_input: int, n_batch: int = 0, n_labels: int = 0, n_latent: int = 10):
        super(VAE, self).__init__()
        print("STABILITY LAST ")
        self.n_latent = n_latent
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent_layers = 1  # not sure what this is for, no usages?

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(n_input, n_latent)
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(n_input, 1)
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(n_latent, n_input)
        self.px_r = torch.nn.Parameter(torch.randn(n_input, ))

    def sample_from_posterior_z(self, x, y=None):
        x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        return z

    def sample_from_posterior_l(self, x):
        x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x):
        return self.inference(x)[3]

    def get_sample_rate(self, x):
        return self.inference(x)[1]

    def inference(self, x):
        x_ = x
        x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_)
        #print(z)
        ql_m, ql_v, library = self.l_encoder(x_)

        # px_r is None here
        px_scale, px_rate, px_dropout = self.decoder('gene', z, library)
        px_r = torch.exp(self.px_r)

        return px_r, px_rate, px_dropout, px_scale, qz_m, qz_v, z, ql_m, ql_v, library

    def forward(self, x, local_l_mean, local_l_var):
        # Parameters for z latent distribution

        px_r, px_rate, px_dropout, px_scale, qz_m, qz_v, z, ql_m, ql_v, library = self.inference(x)
        self.reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)

        # KL Divergence
        #mean = torch.zeros_like(qz_m)
        #scale = torch.ones_like(qz_v)

        #kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        #kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
        kl_divergence_z = 0.5*(qz_m.pow(2) + qz_v - torch.log(1e-8 + qz_v) - 1.).sum(dim=1)
        kl_divergence_l = 0.5 * ((ql_m-local_l_mean).pow(2)/local_l_var
                            + ql_v/local_l_var
                            + torch.log(1e-8 + local_l_var) - torch.log(1e-8 + ql_v) - 1.).sum(dim=1)
        # kl_gauss_z = 0.5 * tf.reduce_sum( \
        #     tf.square(self.qz_m) + self.qz_v - tf.log(1e-8 + self.qz_v) - 1, 1)
        # kl_gauss_l = 0.5 * tf.reduce_sum( \
        #     tf.square(self.ql_m - local_l_mean) / local_l_var \
        #     + self.ql_v / local_l_var \
        #     + tf.log(1e-8 + local_l_var) - tf.log(1e-8 + self.ql_v) - 1, 1)

        kl_divergence = kl_divergence_z

        return self.reconst_loss + kl_divergence_l, kl_divergence
