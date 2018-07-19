# -*- coding: utf-8 -*-
"""Main module."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.distributions import Multinomial, kl_divergence as kl
from torch.distributions import Normal

from scvi.metrics.clustering import get_latents
from scvi.models.vae import VAE


# VAE model
class VADE(VAE):
    def __init__(self, n_input, n_labels, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, n_batch=0,
                 y_prior=None, dispersion="gene", log_variational=True, reconstruction_loss="zinb"):
        super(VADE, self).__init__(n_input, n_batch, n_labels, n_hidden=n_hidden, n_latent=n_latent,
                                   n_layers=n_layers, dropout_rate=dropout_rate, dispersion=dispersion,
                                   log_variational=log_variational, reconstruction_loss=reconstruction_loss)
        self.y_prior = nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels), requires_grad=False
        )
        self.n_latent_layers = 1
        self.n_latent = n_latent

    def initialize_gmm(self, data_loader, use_cuda=True):
        latents, _, _ = get_latents(self, data_loader, use_cuda=use_cuda)
        data = latents[0]  # latents[0]
        self.gmm = GaussianMixture(n_components=self.n_labels, covariance_type='diag')
        self.gmm.fit(data)
        if self.gmm.converged_:
            print("GMM converged")
        else:
            print("GMM didn't converge")

        self.z_clusters = nn.Parameter(
            torch.from_numpy(self.gmm.means_.astype(np.float32)).to(self.y_prior.device)
        )
        self.log_v_clusters = nn.Parameter(
            torch.log(torch.from_numpy(self.gmm.covariances_.astype(np.float32)).to(self.y_prior.device))
        )

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

    def get_latents(self, x, label=None):
        zs = [self.sample_from_posterior_z(x)]
        return zs[::-1]

    def posterior_assignments(self, x):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)

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

        y_prior = self.y_prior.expand(z.size()[0], self.n_labels)

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

        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)

        reconst_loss = self._reconstruction_loss(x, px_rate, px_r, px_dropout, batch_index, y)

        kl_divergence_l += kl(Multinomial(probs=probs), Multinomial(probs=self.y_prior.view(-1, self.n_labels)))

        # , , , probs
        return reconst_loss + kl_normal, kl_divergence_l + kl_multinomial
