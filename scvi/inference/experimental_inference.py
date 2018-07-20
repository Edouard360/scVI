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
    def fit(self, n_epochs=20, lr=1e-3):
        previous_forward = self.model.forward
        self.model.forward = self.model.forward_vade
        super(VadeInference, self).fit(n_epochs=20, lr=1e-3)
        self.model.forward = previous_forward


class GlowInference(Inference):
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset, train_size=0.5, **kwargs):
        super(GlowInference, self).__init__(model, gene_dataset, **kwargs)
        self.data_loaders = TrainTestDataLoaders(gene_dataset, train_size=train_size)
        self.ll('sequential', verbose=True)
        self.model.initialize(to_cuda(self.data_loaders.sample(), use_cuda=self.use_cuda)[0])

    def fit(self, n_epochs=20, lr=1e-4):
        super(GlowInference, self).fit(n_epochs=n_epochs, lr=lr)

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


class GANInference(VariationalInference):
    default_metrics_to_monitor = ['ll'] + ['entropy_batch_mixing']

    def __init__(self, *args, scale=100, warm_up=0,  **kwargs):
        self.scale = scale
        self.warm_up = warm_up
        print("Scale is ", self.scale)
        print("Warm-up is ", self.warm_up)
        super(GANInference, self).__init__(*args, **kwargs)

    def fit(self, n_epochs=20, lr=1e-3, weight_decay=1e-4):
        self.GAN1 = Classifier(self.model.n_latent, n_labels=self.model.n_batch, n_layers=3)
        if self.use_cuda:
            self.GAN1.cuda()
        self.optimizer_GAN = torch.optim.Adam(filter(lambda p: p.requires_grad, self.GAN1.parameters()), lr=lr,
                                              weight_decay=weight_decay)
        super(VariationalInference, self).fit(n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)

    def loss(self, tensors):
        if self.epoch > self.warm_up:  # Leave a warm-up
            sample_batch, _, _, batch_index, _ = tensors
            z = self.model.sample_from_posterior_z(sample_batch, batch_index)
            cls_loss = (self.scale * F.cross_entropy(self.GAN1(z), batch_index.view(-1)))  # Might rather change lr ?
            self.optimizer_GAN.zero_grad()
            cls_loss.backward(retain_graph=True)
            self.optimizer_GAN.step()
        else:
            cls_loss = 0
        return super(GANInference, self).loss(tensors) - cls_loss
