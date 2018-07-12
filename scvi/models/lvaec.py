import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, kl_divergence as kl

from scvi.models import VAE
from scvi.models.base import SemiSupervisedModel
from scvi.models.classifier import Classifier
from scvi.models.modules import DecoderSCVI, LadderDecoder, LadderEncoder, Encoder
from scvi.models.utils import broadcast_labels


class LVAEC(VAE, SemiSupervisedModel):
    '''
    Ladder VAE for classification: multiple layers of stochastic variable
    Instead of having q(z1|z2), we have q(z1|x)
    '''

    def __init__(self, n_input, n_labels, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, n_batch=0,
                 y_prior=None, dispersion="gene", log_variational=True, reconstruction_loss="zinb"):
        super(LVAEC, self).__init__(n_input, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                                    dropout_rate=dropout_rate, n_batch=n_batch, n_labels=n_labels,
                                    dispersion=dispersion, log_variational=log_variational,
                                    reconstruction_loss=reconstruction_loss)

        n_latent_l = [64, 32, 16]

        self.decoder = DecoderSCVI(n_latent_l[0], n_input, n_cat_list=[n_batch], n_layers=n_layers,
                                   n_hidden=n_hidden, dropout_rate=dropout_rate)

        self.y_prior = nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels), requires_grad=False
        )

        self.classifier = Classifier(n_hidden, n_hidden, n_labels=n_labels, n_layers=n_layers,
                                     dropout_rate=dropout_rate)

        self.z_encoder = Encoder(n_input, n_latent_l[0], n_layers=n_layers, n_hidden=n_hidden,
                                 dropout_rate=dropout_rate)
        self.ladder_encoders = nn.ModuleList(
            [LadderEncoder(n_input, n_latent_l[0], n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate)] +
            [LadderEncoder(n_hidden, n_latent, n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate,
                           n_cat_list=[n_labels]) for n_latent in n_latent_l[1:]])
        self.ladder_decoders = nn.ModuleList(
            [LadderDecoder(n_latent_input, n_latent_output, n_cat_list=[n_labels],
                           n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate)
             for (n_latent_input, n_latent_output) in zip(n_latent_l[:0:-1], n_latent_l[-2::-1])]
        )

        assert len(self.ladder_encoders) == len(self.ladder_decoders) + 1  # + 1: for the DecoderSCVI

        self.n_latent_layers = len(self.ladder_encoders)

    def classify(self, x):
        x_ = torch.log(1 + x)
        (_, _, _), q = self.ladder_encoders[0](x_)
        return self.classifier(q)

    def get_latent(self, x, y):
        q = torch.log(1 + x)
        q_list = []
        for i, ladder_encoder in enumerate(self.ladder_encoders):
            (q_m_hat, q_v_hat, z), q = ladder_encoder(q, y)
            q_list += [(q_m_hat, q_v_hat)]

        z_list = [z]
        for ladder_decoder, (q_m_hat, q_v_hat) in zip(self.ladder_decoders, reversed(q_list[:-1])):
            (q_m, q_v, z), (p_m, p_v) = ladder_decoder(z, q_m_hat, q_v_hat, y)
            z_list += [z]
        return z_list[::-1]  # we have sampled in the reverse order

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        is_labelled = False if y is None else True

        x_ = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x_)

        q_list = []
        q_p = []

        # 1 - Encoding
        (q_m_hat, q_v_hat, z_bottom), q_pred = self.ladder_encoders[0](x_)

        ys, q_preds, q_m_hat, q_v_hat = (
            broadcast_labels(
                y, q_pred, q_m_hat, q_v_hat, n_broadcast=self.n_labels
            )
        )
        q_list += [(q_m_hat, q_v_hat)]

        (q_m_hat, q_v_hat, _), q = self.ladder_encoders[1](q_preds, ys)
        q_list += [(q_m_hat, q_v_hat)]

        (q_m_hat, q_v_hat, z), _ = self.ladder_encoders[2](q, ys)
        q_list += [(q_m_hat, q_v_hat)]

        # 1 - Decoding
        q_m_hat, q_v_hat = q_list[1]
        q_p += [self.ladder_decoders[0](z, q_m_hat, q_v_hat, ys)]  # (q_m, q_v, z), (p_m, p_v)
        z = q_p[0][0][2]

        q_m_hat, q_v_hat = q_list[0]
        q_p += [self.ladder_decoders[1](z, q_m_hat, q_v_hat, ys)]  # (q_m, q_v, z), (p_m, p_v)

        kl_divergence = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        q_m_hat, q_v_hat = q_list[-1]
        mean, var = torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat)
        kl_divergence_unweight = \
            kl(Normal(q_m_hat, torch.sqrt(q_v_hat)), Normal(mean, torch.sqrt(var))).sum(dim=1) + \
            sum([(Normal(q_m, torch.sqrt(q_v)).log_prob(z) - Normal(p_m, torch.sqrt(p_v)).log_prob(z)).sum(dim=1)
                 for (q_m, q_v, z), (p_m, p_v) in q_p])

        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z_bottom, library, batch_index)

        reconst_loss = self._reconstruction_loss(x, px_rate, self.px_r, px_dropout, batch_index, y)

        if is_labelled:
            return reconst_loss, kl_divergence + kl_divergence_unweight

        probs = self.classifier(q_pred)  # That is because everything was broadcasted... bad design

        kl_divergence_unweight = kl_divergence_unweight.view(self.n_labels, -1)
        kl_divergence += (kl_divergence_unweight.t() * probs).sum(dim=1)
        kl_divergence += kl(Categorical(probs=probs),
                            Categorical(probs=self.y_prior.repeat(probs.size(0), 1)))
        return reconst_loss, kl_divergence
