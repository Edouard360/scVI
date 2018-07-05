# -*- coding: utf-8 -*-
"""Main module."""
import copy

import numpy as np
import torch
from torch.distributions import Multinomial, kl_divergence as kl
from torch.distributions import Normal

from scvi.metrics.clustering import get_latent_mean
from scvi.metrics.log_likelihood import log_zinb_positive

torch.backends.cudnn.benchmark = True
from scvi.models.vae import VAE
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import math
import torch.nn.functional as F


# VAE model
class VADE(VAE):
    def __init__(self, n_input, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, dispersion="gene",
                 log_variational=True, reconstruction_loss="zinb", n_batch=0, n_labels=0, use_cuda=False):
        super(VADE, self).__init__(n_input, n_batch=n_batch, n_labels=n_labels, use_cuda=use_cuda,
                                   dispersion=dispersion, reconstruction_loss=reconstruction_loss,
                                   log_variational=log_variational)
        self.n_latent = n_latent
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss

        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.n_labels = n_labels

        # First mistake in Max's y_prior is not trainable
        self.y_prior = (1 / self.n_labels) * torch.ones(self.n_labels)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.n_latent_layers = 1
        if self.use_cuda:
            self.cuda()
            self.y_prior = self.y_prior.cuda()

    def initialize_gmm(self, data_loader):
        latents, _, _ = get_latent_mean(self, data_loader)
        data = latents  # latents[0]
        self.gmm = GaussianMixture(n_components=self.n_labels, covariance_type='diag')
        self.gmm.fit(data)
        if self.gmm.converged_:
            print("GMM converged")
        else:
            print("GMM didn't converge")

        self.z_clusters = nn.Parameter(torch.from_numpy(self.gmm.means_.astype(np.float32)))  #
        self.log_v_clusters = nn.Parameter(
            torch.log(torch.from_numpy(self.gmm.covariances_.astype(np.float32))))  # nn.Parameter()
        if self.use_cuda:
            self.cuda()
            self.log_v_clusters = self.log_v_clusters.cuda()
            self.z_clusters = self.z_clusters.cuda()

    def update_parameters(self, indices):
        new_z_clusters = self.z_clusters.data.new_empty(self.z_clusters.data.size())
        new_log_v_clusters = self.log_v_clusters.data.new_empty(self.log_v_clusters.data.size())
        for i in range(self.n_labels):
            new_z_clusters[indices[i][1]] = self.z_clusters.data[indices[i][0]]
            new_log_v_clusters[indices[i][1]] = self.log_v_clusters.data[indices[i][0]]
        self.z_clusters = nn.Parameter(new_z_clusters)
        self.log_v_clusters = nn.Parameter(new_log_v_clusters)

    def classify(self, x):
        return self.posterior_assignments(x)

    def save_state_dict(self):
        self.saved_state_dict = copy.deepcopy(self.state_dict())

    def restart(self):
        self.load_state_dict(self.saved_state_dict)

    def sample_from_posterior_z(self, x, y=None):
        x = torch.log(1 + x)
        # Here we compute as little as possible to have q(z|x)
        qz_m, qz_v, z = self.z_encoder(x)
        if not self.training:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        x = torch.log(1 + x)
        # Here we compute as little as possible to have q(z|x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, y=None, batch_index=None):
        x = torch.log(1 + x)
        z = self.sample_from_posterior_z(x)
        px = self.decoder.px_decoder(z, batch_index)
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_rate(self, x, y=None, batch_index=None):
        x = torch.log(1 + x)
        z = self.sample_from_posterior_z(x)
        library = self.sample_from_posterior_l(x)
        px = self.decoder.px_decoder(z, batch_index)
        return self.decoder.px_scale_decoder(px) * torch.exp(library)

    def sample(self, z):
        return self.px_scale_decoder(z)

    def get_latents(self, x, label=None):
        zs = [self.sample_from_posterior_z(x)]
        return zs[::-1]

    def posterior_assignments(self, x):
        xs = x
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)

        z_clusters = self.z_clusters.t().unsqueeze(0).expand(z.size()[0], self.n_latent, self.n_labels)
        v_clusters = torch.exp(self.log_v_clusters.t()).unsqueeze(0).expand(z.size()[0], self.n_latent, self.n_labels)

        y_prior = self.y_prior.unsqueeze(0).expand(z.size()[0], self.n_labels)

        zs = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_labels)

        torch.sum((-0.5 * torch.log(2 * math.pi * v_clusters + 1e-8)
                   - (zs - z_clusters) ** 2 / (2 * v_clusters)), dim=1)

        p_c_z = (torch.log(y_prior + 1e-8)
                 + Normal(z_clusters, torch.sqrt(v_clusters)).log_prob(zs).sum(dim=1) + 1e-10)
        return F.softmax(p_c_z, dim=-1)

    def forward_vade(self, x, local_l_mean, local_l_var, batch_index=None, y=None):  # same signature as loss
        if not hasattr(self, 'z_clusters'):
            raise ValueError("Need init with GMM")

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        z_clusters = self.z_clusters.t().unsqueeze(0).expand(z.size()[0], self.n_latent, self.n_labels)
        v_clusters = torch.exp(self.log_v_clusters.t()).unsqueeze(0).expand(z.size()[0], self.n_latent, self.n_labels)

        y_prior = self.y_prior.unsqueeze(0).expand(z.size()[0], self.n_labels)

        zs = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_labels)
        qz_m = qz_m.unsqueeze(2).expand(qz_m.size()[0], qz_m.size()[1], self.n_labels)
        qz_v = qz_v.unsqueeze(2).expand(qz_v.size()[0], qz_v.size()[1], self.n_labels)

        torch.sum((-0.5 * torch.log(2 * math.pi * v_clusters + 1e-8)
                   - (zs - z_clusters) ** 2 / (2 * v_clusters)), dim=1)

        p_c_z = (torch.log(y_prior + 1e-8)
                 + Normal(z_clusters, torch.sqrt(v_clusters)).log_prob(zs).sum(dim=1) + 1e-10)
        probs = F.softmax(p_c_z, dim=-1)

        kl_normal = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(z_clusters, torch.sqrt(v_clusters))).sum(dim=1)
        kl_normal = torch.sum(kl_normal * probs, dim=-1)
        kl_multinomial = kl(Multinomial(probs=probs), Multinomial(probs=y_prior))

        reconst_loss = 0

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)

        # # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss += -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)

        kl_divergence_l += kl(Multinomial(probs=probs), Multinomial(probs=self.y_prior.view(-1, self.n_labels)))

        # , , , probs
        return reconst_loss + kl_normal, kl_divergence_l + kl_multinomial
