import numpy as np
import torch

from scvi.dataset.data_loaders import DataLoaders, AlternateSemiSupervisedDataLoaders
from scvi.inference import JointSemiSupervisedVariationalInference, VariationalInference, Inference
from scvi.metrics.classification import compute_accuracy_svc, compute_accuracy_rf
from scvi.metrics.log_likelihood import compute_glow_log_likelihood
from scvi.models import InfoCatVAEC
from scvi.utils import to_cuda


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

    def __init__(self, model, gene_dataset, n_labelled_samples_per_class=10, **kwargs):
        super(GlowInference, self).__init__(model, gene_dataset, **kwargs)
        self.data_loaders = AlternateSemiSupervisedDataLoaders(gene_dataset,
                                                               n_labelled_samples_per_class=n_labelled_samples_per_class)

    def fit(self, n_epochs=20, lr=1e-4):
        self.ll('sequential', verbose=True)
        self.model.initialize(to_cuda(self.data_loaders.sample(), use_cuda=self.use_cuda)[0])
        self.ll('sequential', verbose=True)
        print("starting training")
        super(GlowInference, self).fit(n_epochs=n_epochs, lr=lr)

    def loss(self, tensors):
        # print(self.model.z.grad)
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
        # ys=[]
        # latents=[]
        # for tensor in  self.data_loaders['train']:
        #     sample_batch, _, _, _, y = to_cuda(tensor, use_cuda=self.use_cuda)[0]
        #     ys+=[y]
        #     latents
        # np.array(torch.cat(ys))

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

        return rf_scores  # svc_scores,
