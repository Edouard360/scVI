import collections

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import Linear

from scvi.models.utils import one_hot


class FCLayers(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=128, n_layers=1, dropout_rate=0.1):
        super(FCLayers, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.fc_layers = FCLayers._sequential(layers_dim, dropout_rate=dropout_rate)

    def forward(self, x, *os):
        return self.fc_layers(x)

    @staticmethod
    def create(n_in, n_out, n_cat=0, n_hidden=128, n_layers=1, dropout_rate=0.1, n_mult=10, n_channels=None, scale=1, doubly_linear=False, strange=False):
        if strange:
            return FCStrangeLinear(n_in, n_out, sum(n_cat), n_hidden=n_hidden, n_layers=n_layers,
                                  dropout_rate=dropout_rate)
        if doubly_linear:
            return FCDoublyLinear(n_in, n_out, sum(n_cat), n_hidden=n_hidden, n_layers=n_layers,
                                  dropout_rate=dropout_rate)
        if n_channels is not None:
            return FCLinearOneHot(n_in, n_out, n_channels, n_hidden=n_hidden, n_layers=n_layers,
                                  dropout_rate=dropout_rate)
        if type(n_cat) is int:
            if n_cat == 0:
                return FCLayers(n_in, n_out, n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate)
            else:
                return OneHotFCLayers(n_in, n_out, n_cat=n_cat, n_hidden=n_hidden, n_layers=n_layers,
                                      dropout_rate=dropout_rate, n_mult=n_mult)
        elif type(n_cat) is list:
            return ManyOneHotFCLayers(n_in, n_out, n_cat_list=n_cat,
                                      n_hidden=n_hidden, n_layers=n_layers, dropout_rate=dropout_rate, n_mult=n_mult,
                                      scale=scale)

    @staticmethod
    def _sequential(layers_dim, n_cat=0, dropout_rate=0.1, n_mult=1, n_batch=None, doubly_linear=False):
        def linear(n_in, n_mult, n_cat, n_out):
            if n_batch is None:
                layer = nn.Linear(n_in + n_mult * n_cat, n_out)
            else:
                if not doubly_linear:
                    layer = LinearOneHot(n_in, n_out, n_batch)
                else:
                    layer = DoublyLinear(n_in, n_out, n_batch)
            return layer

        # dropout_rates = [dropout_rate]*len(layers_dim[:-1])
        # if layers_dim[0]==10:
        #     dropout_rates[0] = 0
        # print("DROPOUT RATES ",dropout_rates)

        return nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                nn.Dropout(p=dropout_rate),
                linear(n_in, n_mult, n_cat, n_out),
                nn.BatchNorm1d(n_out, eps=1e-3, momentum=0.99),
                nn.ReLU())) for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))


class OneHotFCLayers(nn.Module):
    def __init__(self, n_in, n_out, n_cat, n_hidden=128, n_layers=1, dropout_rate=0.1, n_mult=10, scale=1):
        super(OneHotFCLayers, self).__init__()
        print("Initialized One Hot with ", n_cat)
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.n_cat = n_cat
        self.n_mult = n_mult
        self.scale = scale
        self.fc_layers = FCLayers._sequential(layers_dim, n_cat=n_cat, dropout_rate=dropout_rate, n_mult=n_mult)

    def forward(self, x, o, *os):
        if o.size(1) != self.n_cat:
            o = one_hot(o, self.n_cat)
        if self.n_mult != 0:
            o = o.repeat(1, self.n_mult) * self.scale
        for layer in self.fc_layers:
            x = layer(torch.cat((x, o), 1))
        return x


class ManyOneHotFCLayers(nn.Module):
    def __init__(self, n_in, n_out, n_cat_list, n_hidden=128, n_layers=1, dropout_rate=0.1, n_mult=10, scale=1):
        super(ManyOneHotFCLayers, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.n_cat_list = n_cat_list
        self.n_mult = n_mult
        self.scale = scale
        self.fc_layers = FCLayers._sequential(layers_dim, n_cat=sum(n_cat_list), dropout_rate=dropout_rate,
                                              n_mult=n_mult)

    def forward(self, x, *os):
        one_hot_os = []
        for i, o in enumerate(os):
            if o is not None and self.n_cat_list[i]:
                one_hot_o = o
                if o.size(1) != self.n_cat_list[i]:
                    one_hot_o = one_hot(o, self.n_cat_list[i]).repeat(1, self.n_mult) * self.scale
                elif o.size(1) == 1 and self.n_cat_list[i] == 1:
                    one_hot_o = o.type(torch.float32) * self.scale
                one_hot_os += [one_hot_o]
        for layer in self.fc_layers:
            x = layer(torch.cat((x,) + tuple(one_hot_os), 1))
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_latent=10, n_cat=0, n_layers=1, dropout_rate=0.1, n_channels=None):
        super(Encoder, self).__init__()
        self.encoder = FCLayers.create(n_in=n_input, n_out=n_hidden, n_cat=n_cat, n_layers=n_layers, n_hidden=n_hidden,
                                       dropout_rate=dropout_rate, n_channels=n_channels)
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        self.var_encoder = nn.Linear(n_hidden, n_latent)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, o=None):
        # Parameters for latent distribution
        q = self.encoder(x, o)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5, 5))  # (computational stability safeguard)
        latent = self.reparameterize(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):
    def __init__(self, n_latent, n_input, n_hidden=128, n_layers=1, dropout_rate=0.1, n_batch=0, n_labels=0,
                 n_channels=None, n_mult=1, scale=1, doubly_linear=False, strange=True):
        super(DecoderSCVI, self).__init__()
        self.n_batch = n_batch
        self.px_decoder = FCLayers.create(n_in=n_latent, n_out=n_hidden, n_layers=n_layers, n_hidden=n_hidden,
                                          dropout_rate=dropout_rate, n_cat=[n_batch, n_labels], n_channels=n_channels,
                                          n_mult=n_mult, scale=scale, doubly_linear=doubly_linear, strange=strange)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_input), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_input)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_input)

    def forward(self, dispersion, z, library, batch_index=None, y=None):
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, batch_index, y)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(torch.clamp(library, max=12)) * px_scale
        if dispersion == "gene-cell":
            px_r = self.px_r_decoder(px)
            return px_scale, px_r, px_rate, px_dropout
        else:  # dispersion == "gene" / "gene-batch" / "gene-label"
            return px_scale, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    def __init__(self, n_latent, n_output, n_cat=0, n_hidden=128, n_layers=1, dropout_rate=0.1, n_mult=1):
        super(Decoder, self).__init__()
        self.decoder = FCLayers.create(n_in=n_latent, n_out=n_hidden, n_cat=n_cat, n_layers=n_layers,
                                       n_hidden=n_hidden, dropout_rate=dropout_rate, n_mult=n_mult)

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x, o=None):
        # Parameters for latent distribution
        p = self.decoder(x, o)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v


class LadderEncoder(Encoder):
    def forward(self, x, o=None):
        q = self.encoder(x, o)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5, 5))
        latent = self.reparameterize(q_m, q_v)
        return (q_m, q_v, latent), q


class LadderDecoder(Decoder):
    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, q_m_hat, q_v_hat, o=None):
        p = self.decoder(x, o)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(torch.clamp(self.var_decoder(p), -5, 5))

        pr1, pr2 = (1 / q_v_hat, 1 / p_v)

        q_m = ((q_m_hat * pr1) + (p_m * pr2)) / (pr1 + pr2)
        q_v = 1 / (pr1 + pr2)

        latent = self.reparameterize(q_m, q_v)
        return (q_m, q_v, latent), (p_m, p_v)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class LinearOneHot(Linear):
    def __init__(self, in_features, out_features, n_batch, bias=True):
        super(LinearOneHot, self).__init__()
        self.n_batch = n_batch
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, n_batch))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, n_batch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, batch_index, y=None):
        batch_index = batch_index.view(-1)
        result = torch.zeros(input.size(0), self.out_features, dtype=input.dtype, device=input.device)
        for b in range(self.n_batch):
            result[batch_index == b] = F.linear(input[batch_index == b], self.weight[:, :, b], self.bias[:, b])
        return result


class DoublyLinear(nn.Module):
    def __init__(self, n_in, n_out, n_batch):
        super(DoublyLinear, self).__init__()
        self.linear_latent = nn.Linear(n_in, n_out)
        self.linear_onehot = nn.Linear(n_batch, n_out)
        self.n_batch = n_batch

    def forward(self, input, batch_index, y=None):
        return (
            F.linear(input, self.linear_latent.weight) +
            F.linear(one_hot(batch_index, self.n_batch), self.linear_onehot.weight) +
            self.linear_onehot.bias + self.linear_latent.bias
        )


class FCStrangeLinear(nn.Module):
    def __init__(self, n_in, n_out, n_batch, n_hidden=128, n_layers=1, dropout_rate=0.1):
        super(FCStrangeLinear, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.n_batch = n_batch
        self.fc_layers = FCLayers._sequential(layers_dim, n_cat=n_batch, dropout_rate=dropout_rate, n_batch=None,
                                              doubly_linear=False)

    def forward(self, x, batch_index, y=None):
        for layers in self.fc_layers:
            for layer in layers:
                x = layer(torch.cat((x,one_hot(batch_index, self.n_batch)),1)) if isinstance(layer, nn.Linear) else layer(x)
        return x

class FCDoublyLinear(nn.Module):
    def __init__(self, n_in, n_out, n_batch, n_hidden=128, n_layers=1, dropout_rate=0.1):
        super(FCDoublyLinear, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.fc_layers = FCLayers._sequential(layers_dim, n_cat=0, dropout_rate=dropout_rate, n_batch=n_batch)

    def forward(self, x, batch_index, y=None):
        for layers in self.fc_layers:
            for layer in layers:
                x = layer(x, batch_index) if isinstance(layer, DoublyLinear) else layer(x)
        return x


class FCLinearOneHot(nn.Module):
    def __init__(self, n_in, n_out, n_batch, n_hidden=128, n_layers=1, dropout_rate=0.1):
        super(FCLinearOneHot, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.fc_layers = FCLayers._sequential(layers_dim, n_cat=0, dropout_rate=dropout_rate, n_batch=n_batch)

    def forward(self, x, batch_index):
        for layers in self.fc_layers:
            for layer in layers:
                x = layer(x, batch_index) if isinstance(layer, LinearOneHot) else layer(x)
        return x
