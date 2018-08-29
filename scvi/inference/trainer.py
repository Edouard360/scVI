import sys
import time
from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch
from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from scvi.inference.posterior import Posterior

torch.set_grad_enabled(False)


import copy

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


plt.switch_backend('agg')


class Trainer:
    r"""The abstract Trainer class for training a PyTorch model and monitoring its statistics. It should be
    inherited at least with a .loss() function to be optimized in the training loop.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :use_cuda: Default: ``True``.
        :metrics_to_monitor: A list of the metrics to monitor. If not specified, will use the
            ``default_metrics_to_monitor`` as specified in each . Default: ``None``.
        :benchmark: if True, prevents statistics computation in the training. Default: ``False``.
        :verbose: If statistics should be displayed along training. Default: ``None``.
        :frequency: The frequency at which to keep track of statistics. Default: ``None``.
        :early_stopping_metric: The statistics on which to perform early stopping. Default: ``None``.
        :save_best_state_metric:  The statistics on which we keep the network weights achieving the best store, and
            restore them at the end of training. Default: ``None``.
        :on: The data_loader name reference for the ``early_stopping_metric`` and ``save_best_state_metric``, that
            should be specified if any of them is. Default: ``None``.
    """
    default_metrics_to_monitor = []

    def __init__(self, model, gene_dataset, use_cuda=True, metrics_to_monitor=None, benchmark=False,
                 verbose=False, frequency=None, weight_decay=1e-6, early_stopping_kwargs=dict(),
                 data_loader_kwargs=dict()):

        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()

        self.data_loader_kwargs = {
            "batch_size": 128,
            "pin_memory": use_cuda
        }
        self.data_loader_kwargs.update(data_loader_kwargs)

        self.weight_decay = weight_decay
        self.benchmark = benchmark
        self.epoch = 0
        self.training_time = 0

        if metrics_to_monitor is not None:
            self.metrics_to_monitor = metrics_to_monitor
        else:
            self.metrics_to_monitor = self.default_metrics_to_monitor

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
            #self.model.double()

        self.frequency = frequency if not benchmark else None
        self.verbose = verbose

        self.history = defaultdict(lambda: [])

    def compute_metrics(self):
        begin = time.time()
        # with torch.set_grad_enabled(False):

            # if self.frequency and (
            #                 self.epoch == 0 or self.epoch == self.n_epochs or (self.epoch % self.frequency == 0)):
            #     self.model.eval()
            #     if self.verbose:
            #         print("\nEPOCH [%d/%d]: " % (self.epoch, self.n_epochs))
            #
            #     for name, posterior in self._posteriors.items():
            #         print_name = ' '.join([s.capitalize() for s in name.split('_')[-2:]])
            #         if hasattr(posterior, 'to_monitor'):
            #             for metric in posterior.to_monitor:
            #                 if self.verbose:
            #                     print(print_name, end=' : ')
            #                 result = getattr(posterior, metric)(verbose=self.verbose)
            #                 self.history[metric + '_' + name] += [result]
            #     self.model.train()
            #     self.compute_metrics_time += time.time() - begin

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        begin = time.time()
        with torch.set_grad_enabled(True):
            self.model.train()

            if params is None:
                params = filter(lambda p: p.requires_grad, self.model.parameters())

            # SGD
            optimizer = torch.optim.Adam(params, lr=lr, eps=eps)#torch.optim.Adam(params, lr=lr, eps=eps) #weight_decay=self.weight_decay,
            self.compute_metrics_time = 0
            self.n_epochs = n_epochs
            self.compute_metrics()

            with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
                # We have to use tqdm this way so it works in Jupyter notebook.
                # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
                for self.epoch in pbar:
                    self.on_epoch_begin()
                    print(self.epoch)
                    pbar.update(1)
                    for tensors_list in self.data_loaders_loop():

                        #print('\n\n')
                        #print(tensors_list[0][0])
                        loss = self.loss(*tensors_list)
                        #print(loss)
                        optimizer.zero_grad()
                        loss.backward()
                        # weight_dropout = self.model.decoder.px_dropout_decoder.weight
                        # bias_dropout = self.model.decoder.px_dropout_decoder.bias
                        # print(weight_dropout.data.t().size())
                        # print(weight_dropout.grad.t())
                        # print(weight_dropout.data.t())
                        #
                        #
                        # print(bias_dropout.data.size())
                        # print(bias_dropout.grad)
                        # print(bias_dropout.data)

                        # print(self.model.z_encoder.encoder.fc_layers[0].weight.data)
                        # print(self.model.z_encoder.encoder.fc_layers[0].weight.grad)
                        optimizer.step()

                    if not self.on_epoch_end():
                        break

            if self.save_best_state_metric is not None:
                self.model.load_state_dict(self.best_state_dict)
                self.compute_metrics()

            self.model.eval()
            self.training_time += (time.time() - begin) - self.compute_metrics_time
            if self.verbose and self.frequency:
                print("\nTraining time:  %i s. / %i epochs" % (int(self.training_time), self.n_epochs))

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        self.compute_metrics()
        return True

    @property
    @abstractmethod
    def posteriors_loop(self):
        pass

    def data_loaders_loop(self):  # returns an zipped iterable corresponding to loss signature
        data_loaders_loop = [self._posteriors[name] for name in self.posteriors_loop] #.train()
        return zip(data_loaders_loop[0], *[cycle(data_loader) for data_loader in data_loaders_loop[1:]])

    def register_posterior(self, name, value):
        name = name.strip('_')
        self._posteriors[name] = value

    def corrupt_posteriors(self, rate=0.1, corruption="uniform"):
        if not hasattr(self.gene_dataset, 'corrupted'):
            self.gene_dataset.corrupt(rate=rate, corruption=corruption)
        for name, posterior in self._posteriors.items():
            self.register_posterior(name, posterior.corrupted())

    def uncorrupt_posteriors(self):
        for name_, posterior in self._posteriors.items():
            self.register_posterior(name_, posterior.uncorrupted())

    def __getattr__(self, name):
        if '_posteriors' in self.__dict__:
            _posteriors = self.__dict__['_posteriors']
            if name.strip('_') in _posteriors:
                return _posteriors[name.strip('_')]

    def __delattr__(self, name):
        if name.strip('_') in self._posteriors:
            del self._posteriors[name.strip('_')]
        else:
            object.__delattr__(self, name)

    def __setattr__(self, name, value):
        if isinstance(value, Posterior):
            name = name.strip('_')
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def train_test(self, model=None, gene_dataset=None, train_size=0.1, test_size=None, seed=0):
        """
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        """
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset
        n = len(gene_dataset)
        n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
        np.random.seed(seed=seed)
        permutation = np.random.permutation(n)
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test:(n_test + n_train)]

        return (
            self.create_posterior(model, gene_dataset, indices=indices_train),
            self.create_posterior(model, gene_dataset, indices=indices_test)
        )

    def create_posterior(self, model=None, gene_dataset=None, shuffle=False, indices=None):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset
        return Posterior(model, gene_dataset, shuffle=shuffle, indices=indices, use_cuda=self.use_cuda,
                         data_loader_kwargs=self.data_loader_kwargs)


class SequentialSubsetSampler(SubsetRandomSampler):
    def __init__(self, indices):
        self.indices = np.sort(indices)

    def __iter__(self):
        return iter(self.indices)


class UnsupervisedTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset, train_size=0.8, test_size=None, kl=None, **kwargs):
        super(UnsupervisedTrainer, self).__init__(model, gene_dataset, **kwargs)
        self.kl = kl
        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set = self.train_test(model, gene_dataset, train_size, test_size)
            self.train_set.to_monitor = ['ll']
            self.test_set.to_monitor = ['ll']

    @property
    def posteriors_loop(self):
        return ['train_set']

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, _, _ = tensors
        reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var)
        loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        return loss

    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, self.epoch / 400)#self.n_epochs)

