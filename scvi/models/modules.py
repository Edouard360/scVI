import collections

import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.distributions import Normal

from scvi.models.utils import one_hot


class FCLayers(nn.Module):
    def __init__(self, n_in, n_out, n_cat_list=[], n_layers=1, n_hidden=128, dropout_rate=0.1):
        super(FCLayers, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]  # n_cat = 1 will be ignored
        self.fc_layers = nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                nn.Linear(n_in + sum(self.n_cat_list), n_out),
                nn.BatchNorm1d(n_out, eps=1e-3, momentum=0.99),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate))) for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))

    def forward(self, x, *cat_list):
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(cat_list), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    x = torch.cat((x, *one_hot_cat_list), 1)
                x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, n_input, n_output, n_cat_list=[], n_layers=1, n_hidden=128, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=dropout_rate)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, *cat_list):
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5, 5))  # (computational stability safeguard)
        latent = self.reparameterize(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):
    def __init__(self, n_input, n_output, n_cat_list=[], n_layers=1, n_hidden=128, dropout_rate=0.1):
        super(DecoderSCVI, self).__init__()
        self.px_decoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                   n_hidden=n_hidden, dropout_rate=dropout_rate)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion, z, library, *cat_list):
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(torch.clamp(library, max=12)) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    def __init__(self, n_input, n_output, n_cat_list=[], n_layers=1, n_hidden=128, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.decoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=dropout_rate)

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x, *cat_list):
        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v


class LadderEncoder(Encoder):
    def forward(self, x, *cat_list):
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5, 5))
        latent = self.reparameterize(q_m, q_v)
        return (q_m, q_v, latent), q


class LadderDecoder(Decoder):
    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, q_m_hat, q_v_hat, *cat_list):
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(torch.clamp(self.var_decoder(p), -5, 5))

        pr1, pr2 = (1 / q_v_hat, 1 / p_v)

        q_m = ((q_m_hat * pr1) + (p_m * pr2)) / (pr1 + pr2)
        q_v = 1 / (pr1 + pr2)

        latent = self.reparameterize(q_m, q_v)
        return (q_m, q_v, latent), (p_m, p_v)


class LinearStatic(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearStatic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)

    def set_parameters(self, gene_dataset):  # forces an invertible linear mapping from z to mu_z
        data = gene_dataset.X.T

        gmm = GaussianMixture(n_components=self.in_features)
        gmm.fit(data)
        numpy_clusters = gmm.predict(data).reshape(-1, 1)
        print([(numpy_clusters == i).mean() for i in range(self.in_features)])

        gene_clusters = torch.from_numpy(numpy_clusters).to(self.weight.device).type(torch.int)
        self.weight = nn.Parameter(one_hot(gene_clusters, self.in_features), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)

    def backward(self, x):
        t = self.weight.transpose(0, 1)
        # inverse = torch.inverse(F.linear(self.weight.T,self.weight).detach())
        return F.linear(x, F.linear(torch.inverse(F.linear(t, t)), self.weight))


class CouplingVAPNEV(nn.Module):  # 2016 - Deep Variational Inference Without Pixel-Wise Reconstruction
    def __init__(self, x_dim, z_dim):
        super(CouplingVAPNEV, self).__init__()
        self.alpha = nn.Parameter(torch.randn(x_dim, 2))  # 2 because there is l_z and m_z
        self.beta_1 = nn.Parameter(torch.randn(x_dim, 2))
        self.beta_2 = nn.Parameter(torch.randn(x_dim, 2))
        self.b = nn.Parameter(torch.randn(x_dim, 2))

        self.l1 = nn.Linear(x_dim, x_dim)
        self.l2 = nn.Linear(z_dim, x_dim)

        # We can predefine two masks for x_dim to be split in [1;x_dim//2] [x_dim//2;x_dim]
        self.mask = torch.tensor([1, 0], dtype=torch.uint8).repeat(x_dim // 2 + 1)[:x_dim]

    def forward(self, x, z, mask_id=1):
        # x1..d xd+1..D
        # Conditional coupling should involve z
        # (2016) paper.
        # However it seems implementation from https://github.com/taesung89/real-nvp/blob/master/real_nvp/nn.py
        # Is only using x

        if mask_id:
            x_ = torch.masked_select(x, self.mask)
        else:
            x_ = torch.masked_select(x, 1 - self.mask)
        l_z = self.l1(x_) * self.l2(z) + self.beta_1[:, 0] * self.l1(x_) \
            + self.beta_2[:, 0] * self.l2(z) + self.b[:, 0]
        m_z = self.l1(x_) * self.l2(z) + self.beta_1[:, 1] * self.l1(x_) \
            + self.beta_2[:, 1] * self.l2(z) + self.b[:, 1]
        # log(det(J = df/dx)) = sum(l_z)
        return l_z, m_z

    def y_x(self, x, z, mask_id=1):
        l_z, m_z = self(x, z, mask_id=mask_id)
        mask = self.mask if mask_id else 1 - self.mask
        return torch.masked_select(x, mask) + torch.masked_select(x * torch.exp(l_z) + m_z, 1 - mask)

    def x_y(self, y, z, mask_id=1):
        l_z, m_z = self(y, z, mask_id=mask_id)
        mask = self.mask if mask_id else 1 - self.mask
        return torch.masked_select(y, mask) + torch.masked_select((y - m_z) * torch.exp(-l_z), 1 - mask)


class CouplingRealNVP(nn.Module):  # 2017 - DENSITY ESTIMATION USING REAL NVP - Google Brain
    def __init__(self, x_dim, z_dim):
        super(CouplingRealNVP, self).__init__()

        self.s = nn.Linear(x_dim, x_dim)
        self.t = nn.Linear(x_dim, x_dim)

        # We can predefine two masks for x_dim to be split in [1;x_dim//2] [x_dim//2;x_dim]
        self.mask = torch.tensor([1, 0], dtype=torch.uint8).repeat(x_dim // 2 + 1)[:x_dim]

    def y_x(self, x, z=None, mask_id=1):
        mask = self.mask if mask_id else 1 - self.mask
        return torch.masked_select(x, mask) + torch.masked_select(x * torch.exp(self.s(x)) + self.t(x), 1 - mask)

    def x_y(self, y, z=None, mask_id=1):
        mask = self.mask if mask_id else 1 - self.mask
        return torch.masked_select(y, mask) + torch.masked_select((y - self.t(y)) * torch.exp(-self.s(y)), 1 - mask)
