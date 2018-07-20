import torch
import torch.nn as nn
from torch.distributions import Normal, Multinomial, kl_divergence as kl

from scvi.metrics.log_likelihood import log_nb_positive, log_zinb_positive
from scvi.models.classifier import Classifier
from scvi.models.modules import Encoder, DecoderSCVI, Decoder
from scvi.models.utils import broadcast_labels


# VAE model - for classification: VAEC
class VAEC_GMM(nn.Module):
    def __init__(self, n_input, n_labels, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, dispersion="gene",
                 log_variational=True, reconstruction_loss="zinb", n_batch=0, y_prior=None, use_cuda=False):
        super(VAEC_GMM, self).__init__()
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.n_labels = 0 if n_labels == 1 else n_labels
        if self.n_labels == 0:
            raise ValueError("VAEC is only implemented for > 1 label dataset")

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, ))

        self.z_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)
        self.decoder_z1_y = Decoder(n_labels, n_latent, n_layers=n_layers)
        self.decoder = DecoderSCVI(n_latent, n_input, n_hidden=n_hidden, n_layers=n_layers,
                                   dropout_rate=dropout_rate, n_batch=n_batch)

        self.y_prior = y_prior if y_prior is not None else (1 / n_labels) * torch.ones(n_labels)
        self.classifier = Classifier(n_latent, n_hidden, self.n_labels, n_layers, dropout_rate)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.y_prior = self.y_prior.cuda()

    def classify(self, x):
        x_ = torch.log(1 + x)
        qz_m, _, z = self.z_encoder(x_)

        if self.training:
            return self.classifier(z)
        else:
            return self.classifier(qz_m)

    def sample_from_posterior_z(self, x, y):
        x = torch.log(1 + x)
        # Here we compute as little as possible to have q(z|x)
        qz_m, qz_v, z = self.z_encoder(x, y)
        return z

    def sample_from_posterior_l(self, x):
        x = torch.log(1 + x)
        # Here we compute as little as possible to have q(z|x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, y=None, batch_index=None):
        x = torch.log(1 + x)
        z = self.sample_from_posterior_z(x, y)
        px = self.decoder.px_decoder(z, batch_index, y)
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_rate(self, x, y=None, batch_index=None):
        x = torch.log(1 + x)
        z = self.sample_from_posterior_z(x, y)
        library = self.sample_from_posterior_l(x)
        px = self.decoder.px_decoder(z, batch_index, y)
        return self.decoder.px_scale_decoder(px) * torch.exp(library)

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        x_ = xs
        if self.log_variational:
            x_ = torch.log(1 + x)
        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)

        # Enumerate choices of label
        ys, xs, batch_index, local_l_var, local_l_mean, z_, qz_m, qz_v = (
            broadcast_labels(
                ys, xs, batch_index, local_l_var, local_l_mean, z, qz_m, qz_v, n_broadcast=self.n_labels
            )
        )

        ql_m, ql_v, library = self.l_encoder(xs)
        pz_m, pz_v = self.decoder_z1_y(ys)

        if self.dispersion == "gene-cell":
            px_scale, self.px_r, px_rate, px_dropout = self.decoder(self.dispersion, z_, library, batch_index)
        elif self.dispersion == "gene":
            px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z_, library, batch_index)

        # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss = -log_zinb_positive(xs, px_rate, torch.exp(self.px_r), px_dropout)
        elif self.reconstruction_loss == 'nb':
            reconst_loss = -log_nb_positive(xs, px_rate, torch.exp(self.px_r))

        loss_z = (- Normal(pz_m, torch.sqrt(pz_v)).log_prob(z_) +
                  Normal(qz_m, torch.sqrt(qz_v)).log_prob(z_)).sum(dim=1)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=-1)

        kl_divergence = loss_z
        if is_labelled:
            return reconst_loss, kl_divergence + kl_divergence_l

        reconst_loss = reconst_loss.view(self.n_labels, -1)
        kl_divergence = kl_divergence.view(self.n_labels, -1)

        probs = self.classifier(z)
        reconst_loss = (reconst_loss.t() * probs).sum(dim=1)
        kl_divergence = (kl_divergence.t() * probs).sum(dim=1)
        kl_divergence += kl(Multinomial(probs=probs), Multinomial(probs=self.y_prior))

        return reconst_loss, kl_divergence + kl_divergence_l
