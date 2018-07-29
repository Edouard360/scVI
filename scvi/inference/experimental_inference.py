import numpy as np
import torch
import torch.nn.functional as F

from scvi.dataset.data_loaders import DataLoaders, TrainTestDataLoaders
from scvi.inference import JointSemiSupervisedVariationalInference, VariationalInference, Inference
from scvi.metrics.classification import compute_accuracy_svc, compute_accuracy_rf
from scvi.metrics.log_likelihood import compute_glow_log_likelihood
from scvi.models import InfoCatVAEC
from scvi.models.classifier import Classifier


class InfoCatInference(JointSemiSupervisedVariationalInference):
    def __init__(self, model, gene_dataset, **kwargs):
        super(InfoCatInference, self).__init__(model, gene_dataset, **kwargs)
        assert isinstance(self.model, InfoCatVAEC)

    def loss(self, tensors_all, tensors_labelled):
        loss = super(InfoCatInference, self).loss(tensors_all, tensors_labelled)

        x, local_l_mean, _, batch_index, _ = tensors_all
        m_loss = torch.mean(self.model.mutual_information_probs(x, local_l_mean=local_l_mean, batch_index=batch_index))
        return loss + m_loss


class VadeInference(VariationalInference):
    def train(self, n_epochs=20, lr=1e-3):
        previous_forward = self.model.forward
        self.model.forward = self.model.forward_vade
        super(VadeInference, self).train(n_epochs=20, lr=1e-3)
        self.model.forward = previous_forward


class GlowInference(Inference):
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset, train_size=0.5, **kwargs):
        super(GlowInference, self).__init__(model, gene_dataset, **kwargs)
        self.data_loaders = TrainTestDataLoaders(gene_dataset, train_size=train_size)
        self.ll('sequential', verbose=True)
        self.model.initialize(to_cuda(self.data_loaders.sample(), use_cuda=self.use_cuda)[0])

    def train(self, n_epochs=20, lr=1e-4):
        super(GlowInference, self).train(n_epochs=n_epochs, lr=lr)

    def loss(self, tensors):
        sample_batch, _, _, _, _ = tensors
        log_p_x = self.model.loss(sample_batch)
        loss = torch.mean(log_p_x)
        return loss

    def ll(self, name, verbose=False):
        ll = compute_glow_log_likelihood(self.model, self.data_loaders[name], use_cuda=self.use_cuda)
        if verbose:
            print("LL for %s is : %.4f" % (name, ll))
        return ll

    ll.mode = 'min'

    def svc_rf(self):
        raw_data = DataLoaders.raw_data(self.data_loaders['labelled'], self.data_loaders['unlabelled'])
        (data_train, labels_train), (data_test, labels_test) = raw_data
        svc_scores = compute_accuracy_svc(data_train, labels_train, data_test, labels_test)
        rf_scores = compute_accuracy_rf(data_train, labels_train, data_test, labels_test)
        print(svc_scores[1])
        print(rf_scores[1])
        print()

        data_train, _ = self.model(torch.from_numpy(data_train.astype(np.float32)).cuda())
        data_test, _ = self.model(torch.from_numpy(data_test.astype(np.float32)).cuda())
        print(data_train.shape)
        data_train = np.array(data_train.detach())
        data_test = np.array(data_test.detach())

        svc_scores = compute_accuracy_svc(data_train, labels_train, data_test, labels_test)
        print(svc_scores[1])
        rf_scores = compute_accuracy_rf(data_train, labels_train, data_test, labels_test)
        print(rf_scores[1])

import types
from math import sqrt, pi

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Uniform

from scvi.models.classifier import Classifier


def mmd_fourier(x1, x2, bandwidth=2., dim_r=500):
    d = x1.size(1)
    rw_n = sqrt(2. / bandwidth) * Normal(0., 1. / sqrt(d)).sample((dim_r, d)).type(x1.type())
    rb_u = 2 * pi * Uniform(0., 1.).sample((dim_r,)).type(x1.type())
    rf0 = sqrt(2. / dim_r) * torch.cos(F.linear(x1, rw_n, rb_u))
    rf1 = sqrt(2. / dim_r) * torch.cos(F.linear(x2, rw_n, rb_u))
    result = (torch.pow(rf0.mean(dim=0) - rf1.mean(dim=0), 2)).sum()
    return torch.sqrt(result)


def mmd_objective(z, batch_index, n_batch):
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


def mmd_loss(self, tensors, *next_tensors):
    if self.epoch > self.warm_up:  # Leave a warm-up
        sample_batch, _, _, batch_index, label = tensors
        qm_z, _, _ = self.model.z_encoder(torch.log(1 + sample_batch), label)  # label only used in VAEC
        loss = mmd_objective(qm_z, batch_index, self.gene_dataset.n_batches)
    else:
        loss = 0
    return type(self).loss(self, tensors, *next_tensors) + loss


def mmd_wrapper(infer, warm_up=100, scale=50):
    infer.warm_up = warm_up
    infer.scale = scale
    infer.loss = types.MethodType(mmd_loss, infer)
    return infer


def adversarial_loss(self, tensors, *next_tensors):
    if self.epoch > self.warm_up:
        sample_batch, _, _, batch_index, label = tensors
        qm_z, _, _ = self.model.z_encoder(torch.log(1 + sample_batch), label)  # label only used in VAEC
        cls_loss = (self.scale * F.cross_entropy(self.adversarial_cls(qm_z), batch_index.view(-1)))
        self.optimizer_cls.zero_grad()
        cls_loss.backward(retain_graph=True)
        self.optimizer_cls.step()
    else:
        cls_loss = 0
    return type(self).loss(self, tensors, *next_tensors) - cls_loss


def adversarial_train(self, n_epochs=20, lr=1e-3, weight_decay=1e-4):
    self.adversarial_cls = Classifier(self.model.n_latent, n_labels=self.model.n_batch, n_layers=3)
    if self.use_cuda:
        self.adversarial_cls.cuda()
    self.optimizer_cls = torch.optim.Adam(filter(lambda p: p.requires_grad, self.adversarial_cls.parameters()), lr=lr,
                                          weight_decay=weight_decay)
    type(self).train(self, n_epochs=n_epochs, lr=lr)


def adversarial_wrapper(infer, warm_up=100, scale=50):
    infer.warm_up = warm_up
    infer.scale = scale
    infer.loss = types.MethodType(adversarial_loss, infer)
    infer.train = types.MethodType(adversarial_train, infer)
    return infer
