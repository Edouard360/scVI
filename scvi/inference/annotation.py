import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from scvi.inference import Trainer
from scvi.inference.inference import UnsupervisedTrainer
from scvi.inference.posterior import compute_accuracy_classifier, compute_accuracy_tuple
import torch.nn as nn

def eval_batch_norm(module):
    children = [c for c in module.children()]
    while len(children):
        child = children.pop()
        children += [c for c in child.children()]
        if isinstance(child, nn.BatchNorm1d):
            child.eval()


class ClassifierTrainer(Trainer):
    r"""The ClassifierInference class for training a classifier either on the raw data or on top of the latent
        space of another model (VAE, VAEC, SCANVI).

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
            to use Default: ``0.8``.
        :\**kwargs: Other keywords arguments from the general Trainer class.


    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> classifier = Classifier(vae.n_latent, n_labels=cortex_dataset.n_labels)
        >>> trainer = ClassifierTrainer(classifier, gene_dataset, sampling_model=vae, train_size=0.5)
        >>> trainer.train(n_epochs=20, lr=1e-3)
        >>> trainer.test_set.accuracy()
    """

    def __init__(self, *args, sampling_model=None, use_cuda=True, **kwargs):
        self.sampling_model = sampling_model
        super(ClassifierTrainer, self).__init__(*args, use_cuda=use_cuda, **kwargs)
        self.train_set, self.test_set = self.train_test(self.model, self.gene_dataset)

    @property
    def posteriors_loop(self):
        return ['train_set']

    @property
    def train_set(self):
        return self._train_set

    @train_set.setter
    def train_set(self, train_set):
        if self.sampling_model is not None:
            train_set.sampling_model = self.sampling_model
        self._train_set = train_set

    @property
    def test_set(self):
        return self._test_set

    @test_set.setter
    def test_set(self, test_set):
        if self.sampling_model is not None:
            test_set.sampling_model = self.sampling_model
        self._test_set = test_set

    def loss(self, tensors_labelled):
        x, _, _, _, labels_train = tensors_labelled
        x = self.sampling_model.z_encoder(torch.log(1+x))[0] if self.sampling_model is not None else x
        return F.cross_entropy(self.model(x), labels_train.view(-1))



class SemiSupervisedTrainer(UnsupervisedTrainer):
    r"""The SemiSupervisedTrainer class for the semi-supervised training of an autoencoder.
    This parent class can be inherited to specify the different training schemes for semi-supervised learning
    """

    def __init__(self, model, gene_dataset, n_labelled_samples_per_class=50, n_epochs_classifier=1,
                 lr_classification=0.1, classification_ratio=1, seed=0, **kwargs):
        """
        :param n_labelled_samples_per_class: number of labelled samples per class
        """
        super(SemiSupervisedTrainer, self).__init__(model, gene_dataset, **kwargs)
        self.model = model
        self.gene_dataset = gene_dataset

        self.n_epochs_classifier = n_epochs_classifier
        self.lr_classification = lr_classification
        self.classification_ratio = classification_ratio
        n_labelled_samples_per_class_array = [n_labelled_samples_per_class] * self.gene_dataset.n_labels
        labels = np.array(self.gene_dataset.labels).ravel()
        np.random.seed(seed=seed)
        permutation_idx = np.random.permutation(len(labels))
        labels = labels[permutation_idx]
        indices = []
        current_nbrs = np.zeros(len(n_labelled_samples_per_class_array))
        for idx, (label) in enumerate(labels):
            label = int(label)
            if current_nbrs[label] < n_labelled_samples_per_class_array[label]:
                indices.insert(0, idx)
                current_nbrs[label] += 1
            else:
                indices.append(idx)
        indices = np.array(indices)
        total_labelled = sum(n_labelled_samples_per_class_array)
        indices_labelled = permutation_idx[indices[:total_labelled]]
        indices_unlabelled = permutation_idx[indices[total_labelled:]]

        self.classifier_trainer = ClassifierTrainer(
            model.classifier, gene_dataset, metrics_to_monitor=[], verbose=True, frequency=0,
            sampling_model=self.model
        )

        self.full_dataset = self.create_posterior(shuffle=True)
        self.labelled_set = self.create_posterior(indices=indices_labelled)
        self.unlabelled_set = self.create_posterior(indices=indices_unlabelled)

        self.classifier_trainer.train_set = self.labelled_set

        for posterior in [self.labelled_set, self.unlabelled_set]:
            posterior.to_monitor = ['ll', 'accuracy']

    @property
    def posteriors_loop(self):
        return ['full_dataset', 'labelled_set']

    @property
    def labelled_set(self):
        return self._labelled_set

    @labelled_set.setter
    def labelled_set(self, labelled_set):
        self._labelled_set = labelled_set
        self.classifier_trainer.train_set = labelled_set

    def loss(self, tensors_all, tensors_labelled):
        loss = super(SemiSupervisedTrainer, self).loss(tensors_all)
        eval_batch_norm(self.model)

        sample_batch, _, _, _, y = tensors_labelled
        classification_loss = F.cross_entropy(self.model.classify(sample_batch), y.view(-1))
        loss += classification_loss * self.classification_ratio
        self.model.train()
        return loss

    def on_epoch_end(self):
        eval_batch_norm(self.model)
        self.classifier_trainer.train(self.n_epochs_classifier, lr=self.lr_classification)
        self.model.train()
        return super(SemiSupervisedTrainer, self).on_epoch_end()

    def nn_latentspace(self, verbose=False):
        data_train, _, labels_train = self.labelled_set.get_latent()
        data_test, _, labels_test = self.unlabelled_set.get_latent()
        nn = KNeighborsClassifier()
        nn.fit(data_train, labels_train)
        score = nn.score(data_test, labels_test)
        if verbose:
            print("NN classifier score:", score)
            print("NN classifier tuple:", compute_accuracy_tuple(labels_test, nn.predict(data_test)))
        return score


class JointSemiSupervisedTrainer(SemiSupervisedTrainer):
    def __init__(self, model, gene_dataset, **kwargs):
        kwargs.update({'n_epochs_classifier': 0})
        super(JointSemiSupervisedTrainer, self).__init__(model, gene_dataset, **kwargs)


class AlternateSemiSupervisedTrainer(SemiSupervisedTrainer):
    def __init__(self, *args, **kwargs):
        super(AlternateSemiSupervisedTrainer, self).__init__(*args, **kwargs)

    def loss(self, all_tensor):
        return super(SemiSupervisedTrainer, self).loss(all_tensor)

    @property
    def posteriors_loop(self):
        return ['full_dataset']


class SemiSupervisedTrainerKnn(SemiSupervisedTrainer):
    def __init__(self, *args, frequency_knn=10, knn_classification_ratio=50, **kwargs):
        super(SemiSupervisedTrainerKnn, self).__init__(*args, **kwargs)
        self.frequency_knn = frequency_knn
        self.knn_classification_ratio = knn_classification_ratio

        self.classifier_trainer = ClassifierTrainer(
            self.model.classifier, self.gene_dataset, metrics_to_monitor=[], verbose=True, frequency=0,
            sampling_model=self.model
        )
        self.classifier_trainer.labelled_set = self.labelled_set

    def update_unlabelled_knn(self):
        with torch.set_grad_enabled(False):
            self.model.eval()
            latent_train, _, labels_train = self.labelled_set.get_latent()
            latent_test, _, labels_test = self.unlabelled_set.sequential().get_latent()  # need original counts as input
            counts_test, _ = self.unlabelled_set.sequential().raw_data()
            nn = KNeighborsClassifier()
            # on a subset of the train data restriction
            # latent_train = latent_train
            # labels_train = labels_train
            nn.fit(latent_train, labels_train)
            print("SCORE nn :", nn.score(latent_test, labels_test))

            proba_test = nn.predict_proba(latent_test)
            classification_ratio = np.zeros((len(latent_test), self.gene_dataset.n_labels))
            classification_ratio[:, nn.classes_] = proba_test

            to_keep = proba_test.max(axis=1) >= 0.8  # Threshold of confidence

            # # 1 - Maxime's initial formulation
            # classification_ratio = np.log(classification_ratio + 1e-8) * self.knn_classification_ratio
            # # 2 - Alternative here classification_ratio > 0 implies the opposite sign somewher
            # classification_ratio = -np.log(1 - classification_ratio)

            counts_test, labels_test, classification_ratio = \
                counts_test[to_keep], labels_test[to_keep], classification_ratio[to_keep]

            labelled_test_set = TensorDataset(torch.from_numpy(counts_test.astype(np.float32)),
                                              torch.from_numpy(labels_test.astype(np.int64)),
                                              torch.from_numpy(classification_ratio.astype(np.float32)))

            self.unlabelled_knn_set = self.create_posterior(gene_dataset=labelled_test_set, shuffle=True)
            self.model.train()

    @property
    def posteriors_loop(self):
        return ['full_dataset', 'labelled_set', 'unlabelled_knn_set']

    @property
    def unlabelled_knn_set(self):
        return self._unlabelled_knn_set

    @unlabelled_knn_set.setter
    def unlabelled_knn_set(self, unlabelled_knn_set):
        self._unlabelled_knn_set = unlabelled_knn_set
        self.classifier_trainer.unlabelled_knn_set = unlabelled_knn_set

    def loss(self, tensors_all, tensors_labelled, tensors_unlabelled_knn):
        loss = super(SemiSupervisedTrainerKnn, self).loss(tensors_all, tensors_labelled)
        sample_batch, labels_knn, classification_ratio = tensors_unlabelled_knn

        # if classification_ratio < 0, we decrease prediction for high uncertainty (high classification_ratio value)
        # loss_classification = - torch.sum(self.model.classify(sample_batch) * classification_ratio)

        loss_classification = self.knn_classification_ratio * \
                              F.cross_entropy(self.model.classify(sample_batch), labels_knn.view(-1))
        return loss + loss_classification

    def on_epoch_begin(self):
        if self.epoch % self.frequency_knn == 0:
            print("Updating unlabelled data_loader with KNN")
            self.update_unlabelled_knn()
        super(SemiSupervisedTrainerKnn, self).on_epoch_begin()


def compute_accuracy_svc(data_train, labels_train, data_test, labels_test, param_grid=None, verbose=0, max_iter=-1):
    if param_grid is None:
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
        ]
    svc = SVC(max_iter=max_iter)
    clf = GridSearchCV(svc, param_grid, verbose=verbose)
    return compute_accuracy_classifier(clf, data_train, labels_train, data_test, labels_test)


def compute_accuracy_rf(data_train, labels_train, data_test, labels_test, param_grid=None, verbose=0):
    if param_grid is None:
        param_grid = {'max_depth': np.arange(3, 10), 'n_estimators': [10, 50, 100, 200]}
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    clf = GridSearchCV(rf, param_grid, verbose=verbose)
    return compute_accuracy_classifier(clf, data_train, labels_train, data_test, labels_test)
