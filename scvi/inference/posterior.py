import copy
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from scipy.stats import kde, entropy, itemfreq
from sklearn import neighbors
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.distributions import Normal, kl_divergence as kl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler, RandomSampler
from scvi.models.vae import log_zinb_positive
from scvi.models.vae import compute_log_likelihood


class SequentialSubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)

# class DataLoaderIter:
#     def __init__(self, dataset):
#         batch_size = 128
#         self.dataset = dataset
#         iterep = int((len(self.dataset) - 1) / float(batch_size)) + 1
#         self.index_train_list = iter([np.random.choice(np.arange(len(self.dataset)), size=batch_size)
#                                       for t in range(iterep)])
#
#     def __next__(self):
#         return self.dataset.collate_fn(next(self.index_train_list))
#
#     def __iter__(self):
#         return self
#
#
# class MyDataLoader(DataLoader):
#     def __iter__(self):
#         return DataLoaderIter(self.dataset)


class Posterior:
    def __init__(self, model, gene_dataset, shuffle=False, indices=None, use_cuda=True, data_loader_kwargs=dict()):
        self.model = model
        self.gene_dataset = gene_dataset
        self.to_monitor = []
        self.use_cuda = use_cuda

        if indices is not None and shuffle:
            raise ValueError('indices is mutually exclusive with shuffle')
        if indices is None:
            if shuffle:
                sampler = RandomSampler(gene_dataset)
            else:
                sampler = SequentialSampler(gene_dataset)
        else:
            if hasattr(indices, 'dtype') and indices.dtype is np.dtype('bool'):
                indices = np.where(indices)[0].ravel()
            sampler = SubsetRandomSampler(indices)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        if hasattr(gene_dataset, 'collate_fn'):
            self.data_loader_kwargs.update({'collate_fn': gene_dataset.collate_fn})
        self.data_loader_kwargs.update({'sampler': sampler})
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    # def train(self):
    #     return self.update(dict(), data_loader_type=MyDataLoader)

    @property
    def indices(self):
        if hasattr(self.data_loader.sampler, 'indices'):
            return self.data_loader.sampler.indices
        else:
            return np.arange(len(self.gene_dataset))

    def __iter__(self):
        return map(self.to_cuda, iter(self.data_loader))

    def to_cuda(self, tensors):
        return [t.cuda(async=self.use_cuda) if self.use_cuda else t for t in tensors]

    def update(self, data_loader_kwargs, data_loader_type=DataLoader):
        posterior = copy.copy(self)
        posterior.data_loader_kwargs = copy.copy(self.data_loader_kwargs)
        posterior.data_loader_kwargs.update(data_loader_kwargs)
        posterior.data_loader = data_loader_type(self.gene_dataset, **posterior.data_loader_kwargs)
        return posterior

    def sequential(self, batch_size=128):
        return self.update({'batch_size': batch_size, 'sampler': SequentialSubsetSampler(indices=self.indices)})

    def corrupted(self):
        return self.update({'collate_fn': self.gene_dataset.collate_fn_corrupted})

    def uncorrupted(self):
        return self.update({'collate_fn': self.gene_dataset.collate_fn})

    def ll(self, verbose=False):
        ll = compute_log_likelihood(self.model, self)
        if verbose:
            print("LL : %.4f" % ll)
        return ll

    ll.mode = 'min'

    def get_latent(self):
        latent = []
        batch_indices = []
        labels = []
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            latent += [self.model.sample_from_posterior_z(sample_batch, y=label)]
            batch_indices += [batch_index]
            labels += [label]
        return np.array(torch.cat(latent)), np.array(torch.cat(batch_indices)), np.array(torch.cat(labels)).ravel()

    def entropy_batch_mixing(self, verbose=False, **kwargs):
        if self.gene_dataset.n_batches == 2:
            latent, batch_indices, labels = self.get_latent()
            be_score = entropy_batch_mixing(latent, batch_indices, **kwargs)
            if verbose:
                print("Entropy batch mixing :", be_score)
            return be_score

    entropy_batch_mixing.mode = 'max'

    def differential_expression_stats(self, M_sampling=100):
        """
        Output average over statistics in a symmetric way (a against b)
        forget the sets if permutation is True
        :param vae: The generative vae and encoder network
        :param data_loader: a data loader for a particular dataset
        :param M_sampling: number of samples
        :return: A 1-d vector of statistics of size n_genes
        """
        px_scales = []
        all_labels = []
        batch_size = max(self.data_loader_kwargs['batch_size'] // M_sampling, 2)  # Reduce batch_size on GPU
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            px_scales += [
                np.array((self.model.get_sample_scale(
                    sample_batch, batch_index=batch_index, y=labels, n_samples=M_sampling)
                         ).cpu())]

            # Align the sampling
            px_scales[-1] = (px_scales[-1].transpose((1, 0, 2))).reshape(-1, px_scales[-1].shape[-1])
            all_labels += [np.array((labels.repeat(1, M_sampling).view(-1, 1)).cpu())]

        px_scales = np.concatenate(px_scales)
        all_labels = np.concatenate(all_labels).ravel()  # this will be used as boolean

        return px_scales, all_labels

    def differential_expression_score(self, cell_type, other_cell_type=None, genes=None, M_sampling=100,
                                      M_permutation=10000, permutation=False):
        px_scale, all_labels = self.differential_expression_stats(M_sampling=M_sampling)
        cell_idx, other_cell_idx = self.gene_dataset._cell_type_idx([cell_type, other_cell_type])
        if genes is not None:
            px_scale = px_scale[:, self.gene_dataset._gene_idx(genes)]
        bayes_factors_list = get_bayes_factors(px_scale, all_labels, cell_idx, other_cell_idx=other_cell_idx,
                                               M_permutation=M_permutation, permutation=permutation)
        return bayes_factors_list

    def differential_expression_table(self, output_file=False, select=10, M_sampling=100, M_permutation=10000,
                                      permutation=False):
        px_scale, all_labels = self.differential_expression_stats(M_sampling=M_sampling)
        expression = []
        genes = []
        gene_names = self.gene_dataset.gene_names
        n_cells = self.gene_dataset.n_labels
        for cell_idx in range(n_cells):
            bayes_factors_list = get_bayes_factors(px_scale, all_labels, cell_idx, M_permutation=M_permutation,
                                                   permutation=permutation)
            expression.append(bayes_factors_list)
            top_indices = np.argsort(bayes_factors_list)[::-1][:select]
            top_genes = gene_names[top_indices]
            genes += list(top_genes)

        expression = [[res[np.where(gene_names == gene_name)[0]][0] for gene_name in genes] for res in expression]
        genes = np.array(genes)
        expression = np.array(expression).T  # change to genes * clusters
        if output_file:  # store as an excel spreadsheet
            writer = pd.ExcelWriter('data/differential_expression.xlsx', engine='xlsxwriter')
            clusters = self.gene_dataset.cell_types
            for cluster_idx in range(len(clusters)):
                df = pd.DataFrame(data=expression[select * cluster_idx: select * (cluster_idx + 1), cluster_idx],
                                  index=genes[select * cluster_idx: select * (cluster_idx + 1)],
                                  columns=['differential_expression'])
                df.to_excel(writer, sheet_name=clusters[cluster_idx])
            writer.close()
        return genes, expression

    def generate(self, n_samples=100, genes=None):  # with n_samples>1 return original list/ otherwose sequential
        original_list = []
        posterior_list = []
        batch_size = 128  # max(self.data_loader_kwargs['batch_size'] // n_samples, 2)  # Reduce batch_size on GPU
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            px_dispersion, px_rate = self.model.inference(sample_batch, batch_index=batch_index, y=labels,
                                                          n_samples=n_samples)[:2]

            p = px_rate / (px_rate + px_dispersion)
            r = px_dispersion
            #
            l_train = np.random.gamma(r, p / (1 - p))
            X = np.random.poisson(l_train)
            # '''
            # In numpy (shape, scale) => (concentration, rate), with scale = p /(1 - p)
            # rate = (1 - p) / p  # = 1/scale # used in pytorch
            # l_train = Gamma(r, rate).sample()  # assert Gamma(r, rate).mean = px_rate
            # posterior = Poisson(l_train).sample()
            # '''
            original_list += [np.array(sample_batch.cpu())]
            posterior_list += [X]  # [np.array(posterior.cpu())]##

            if genes is not None:
                posterior_list[-1] = posterior_list[-1][:, :, self.gene_dataset._gene_idx(genes)]
                original_list[-1] = original_list[-1][:, self.gene_dataset._gene_idx(genes)]

            posterior_list[-1] = np.transpose(posterior_list[-1], (1, 2, 0))

        return np.concatenate(posterior_list, axis=0), np.concatenate(original_list, axis=0)

    def generate_parameters(self):
        dropout_list = []
        mean_list = []
        dispersion_list = []
        for tensors in self.sequential(1000):
            sample_batch, _, _, batch_index, labels = tensors
            px_dispersion, px_rate, px_dropout = self.model.inference(sample_batch)[:3]

            dispersion_list += [np.repeat(np.array(px_dispersion.cpu())[np.newaxis, :], px_rate.size(0), axis=0)]
            mean_list += [np.array(px_rate.cpu())]
            dropout_list += [np.array(px_dropout.cpu())]

        return np.concatenate(dropout_list), np.concatenate(mean_list), np.concatenate(dispersion_list)


    def get_stats(self, verbose=True):
        kl_divergence_l_list = []
        kl_divergence_z_list = []
        reconst_losses = []
        qz_v_list = []
        for tensors in self:
            x, local_l_mean, local_l_var, batch_index, y = tensors
            px_r, px_rate, px_dropout, px_scale, qz_m, qz_v, z, ql_m, ql_v, library = self.model.inference(x)
            qz_v_list += [qz_v]


            reconst_losses += [np.array((-log_zinb_positive(x, px_rate, px_r, px_dropout)).cpu())]
            # print(reconst_losses[-1].shape)
            # KL Divergence
            mean = torch.zeros_like(qz_m)
            scale = torch.ones_like(qz_v)

            kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
            kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(
                dim=1)
            kl_divergence_z_list += [np.array(kl_divergence_z.cpu())]
            kl_divergence_l_list += [np.array(kl_divergence_l.cpu())]

        # print(len())
        reconst_losses = np.concatenate(reconst_losses)
        kl_divergence_z_list = np.concatenate(kl_divergence_z_list)
        kl_divergence_l_list = np.concatenate(kl_divergence_l_list)
        qz_v_list = np.concatenate(qz_v_list)

        print({'recons': np.mean(reconst_losses), 'mean_kl_z': np.mean(kl_divergence_z_list),
               'max_kl_z': np.max(kl_divergence_z_list),
               'mean_kl_l': np.mean(kl_divergence_l_list), 'mean_qz_v': np.mean(qz_v_list),
               'min_qz_v': np.min(qz_v_list)})
        return None

    def get_library(self, verbose=True):
        library_list = []
        for tensors in self:
            x = tensors[0]
            library_list += [np.array(self.model.inference(x)[-1].cpu())]
        library_list = np.concatenate(library_list)
        return library_list

    def get_sample_scale(self):
        px_scales = []
        for tensors in self:
            sample_batch, _, _, batch_index, labels = tensors
            px_scales += [
                np.array((self.model.get_sample_scale(sample_batch)).cpu())]
        return np.concatenate(px_scales)

    def show_t_sne(self, n_samples=1000, color_by='', save_name='', latent=None, batch_indices=None,
                   labels=None, n_batch=None):
        # If no latent representation is given
        if latent is None:
            latent, batch_indices, labels = self.get_latent()
            latent, idx_t_sne = self.apply_t_sne(latent, n_samples)
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
        if not color_by:
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1])
        if color_by == 'scalar':
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1], c=labels.ravel())
        else:
            if n_batch is None:
                n_batch = self.gene_dataset.n_batches
            if color_by == 'batches' or color_by == 'labels':
                indices = batch_indices if color_by == 'batches' else labels
                n = n_batch if color_by == 'batches' else self.gene_dataset.n_labels
                if hasattr(self.gene_dataset, 'cell_types') and color_by == 'labels':
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                plt.figure(figsize=(10, 10))
                for i, label in zip(range(n), plt_labels):
                    plt.scatter(latent[indices == i, 0], latent[indices == i, 1], label=label)
                plt.legend()
            elif color_by == 'batches and labels':
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                for i in range(n_batch):
                    axes[0].scatter(latent[batch_indices == i, 0], latent[batch_indices == i, 1], label=str(i))
                axes[0].set_title("batch coloring")
                axes[0].axis("off")
                axes[0].legend()

                indices = labels
                if hasattr(self.gene_dataset, 'cell_types'):
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                for i, cell_type in zip(range(self.gene_dataset.n_labels), plt_labels):
                    axes[1].scatter(latent[indices == i, 0], latent[indices == i, 1], label=cell_type)
                axes[1].set_title("label coloring")
                axes[1].axis("off")
                axes[1].legend()
        plt.axis("off")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)

    @staticmethod
    def apply_t_sne(latent, n_samples=1000):
        idx_t_sne = np.random.permutation(len(latent))[:n_samples] if n_samples else np.arange(len(latent))
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        return latent, idx_t_sne

    def raw_data(self):
        """
        Returns raw data for classification
        """
        return self.gene_dataset.X[self.indices], self.gene_dataset.labels[self.indices].ravel()


def entropy_from_indices(indices):
    return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))


def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100, max_number=500):
    # max number is an important parameter
    n_samples = len(latent_space)
    keep_idx = np.random.choice(np.arange(n_samples), size=min(len(latent_space), max_number), replace=False)
    latent_space, batches = latent_space[keep_idx], batches[keep_idx]

    batches = batches.ravel()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent_space)
    indices = nbrs.kneighbors(latent_space, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: batches[i])(indices)
    entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score


def get_bayes_factors(px_scale, all_labels, cell_idx, other_cell_idx=None, genes_idx=None,
                      M_permutation=10000, permutation=False):
    '''
    Returns a list of bayes factor for all genes
    :param px_scale: The gene frequency array for all cells (might contain multiple samples per cells)
    :param all_labels: The labels array for the corresponding cell types
    :param cell_idx: The first cell type population to consider. Either a string or an idx
    :param other_cell_idx: (optional) The second cell type population to consider. Either a string or an idx
    :param M_permutation: The number of permuted samples.
    :param permutation: Whether or not to permute.
    :return:
    '''
    idx = (all_labels == cell_idx)
    idx_other = (all_labels == other_cell_idx) if other_cell_idx is not None else (all_labels != other_cell_idx)
    if genes_idx is not None:
        px_scale = px_scale[:, genes_idx]
    sample_rate_a = px_scale[idx].reshape(-1, px_scale.shape[1])
    sample_rate_b = px_scale[idx_other].reshape(-1, px_scale.shape[1])

    # agregate dataset
    samples = np.vstack((sample_rate_a, sample_rate_b))

    # prepare the pairs for sampling
    list_1 = list(np.arange(sample_rate_a.shape[0]))
    list_2 = list(sample_rate_a.shape[0] + np.arange(sample_rate_b.shape[0]))
    if not permutation:
        # case1: no permutation, sample from A and then from B
        u, v = np.random.choice(list_1, size=M_permutation), np.random.choice(list_2, size=M_permutation)
    else:
        # case2: permutation, sample from A+B twice
        u, v = (np.random.choice(list_1 + list_2, size=M_permutation),
                np.random.choice(list_1 + list_2, size=M_permutation))

    # then constitutes the pairs
    first_set = samples[u]
    second_set = samples[v]

    res = np.mean(first_set >= second_set, 0)
    res = np.log(res + 1e-8) - np.log(1 - res + 1e-8)
    return res


def plot_imputation(original, imputed, title="Imputation"):
    y = imputed
    x = original

    ymax = 10
    mask = x < ymax
    x = x[mask]
    y = y[mask]

    mask = y < ymax
    x = x[mask]
    y = y[mask]

    l_minimum = np.minimum(x.shape[0], y.shape[0])

    x = x[:l_minimum]
    y = y[:l_minimum]

    data = np.vstack([x, y])

    plt.figure(figsize=(5, 5))

    axes = plt.gca()
    axes.set_xlim([0, ymax])
    axes.set_ylim([0, ymax])

    nbins = 50

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:ymax:nbins * 1j, 0:ymax:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.title(title, fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel('Original counts')

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

    a, _, _, _ = np.linalg.lstsq(y[:, np.newaxis], x, rcond=-1)
    linspace = np.linspace(0, ymax)
    plt.plot(linspace, a * linspace, color='black')

    plt.plot(linspace, linspace, color='black', linestyle=":")
    plt.show()
    plt.savefig(title + '.png')


def nn_overlap(X1, X2, k=100):
    '''
    Compute the overlap between the k-nearest neighbor graph of X1 and X2 using Spearman correlation of the
    adjacency matrices.
    '''
    nne = NearestNeighbors(n_neighbors=k + 1, n_jobs=8)
    assert len(X1) == len(X2)
    n_samples = len(X1)
    nne.fit(X1)
    kmatrix_1 = nne.kneighbors_graph(X1) - scipy.sparse.identity(n_samples)
    nne.fit(X2)
    kmatrix_2 = nne.kneighbors_graph(X2) - scipy.sparse.identity(n_samples)

    # 1 - spearman correlation from knn graphs
    spearman_correlation = scipy.stats.spearmanr(kmatrix_1.A.flatten(), kmatrix_2.A.flatten())[0]
    # 2 - fold enrichment
    set_1 = set(np.where(kmatrix_1.A.flatten() == 1)[0])
    set_2 = set(np.where(kmatrix_2.A.flatten() == 1)[0])
    fold_enrichment = len(set_1.intersection(set_2)) * n_samples ** 2 / (float(len(set_1)) * len(set_2))
    return spearman_correlation, fold_enrichment


def unsupervised_clustering_accuracy(y, y_pred):
    """
    Unsupervised Clustering Accuracy
    """
    assert len(y_pred) == len(y)
    u = np.unique(np.concatenate((y, y_pred)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    cost_matrix = reward_matrix.max() - reward_matrix
    ind = linear_assignment(cost_matrix)
    return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind


def knn_purity(latent, label, n_neighbors=30):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: label[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - label.reshape(-1, 1)) == 0).mean(axis=1)
    res = [np.mean(scores[label == i]) for i in np.unique(label)]  # per cell-type purity

    return np.mean(res)


Accuracy = namedtuple('Accuracy', ['unweighted', 'weighted', 'worst', 'accuracy_classes'])


def compute_accuracy_tuple(y, y_pred):
    y = y.ravel()
    n_labels = len(np.unique(y))
    classes_probabilities = []
    accuracy_classes = []
    for cl in range(n_labels):
        idx = y == cl
        classes_probabilities += [np.mean(idx)]
        accuracy_classes += [np.mean((y[idx] == y_pred[idx])) if classes_probabilities[-1] else 0]
        # This is also referred to as the "recall": p = n_true_positive / (n_false_negative + n_true_positive)
        # ( We could also compute the "precision": p = n_true_positive / (n_false_positive + n_true_positive) )
        accuracy_named_tuple = Accuracy(
            unweighted=np.dot(accuracy_classes, classes_probabilities),
            weighted=np.mean(accuracy_classes),
            worst=np.min(accuracy_classes),
            accuracy_classes=accuracy_classes)
    return accuracy_named_tuple


def compute_predictions(model, data_loader, classifier=None):
    all_y_pred = []
    all_y = []

    for i_batch, tensors in enumerate(data_loader):
        sample_batch, _, _, _, labels = tensors
        all_y += [labels.view(-1)]

        if hasattr(model, 'classify'):
            y_pred = model.classify(sample_batch).argmax(dim=-1)
        elif classifier is not None:
            # Then we use the specified classifier
            if model is not None:
                sample_batch, _, _ = model.z_encoder(sample_batch)
            y_pred = classifier(sample_batch).argmax(dim=-1)
        else:  # The model is the raw classifier
            y_pred = model(sample_batch).argmax(dim=-1)
        all_y_pred += [y_pred]

    all_y_pred = np.array(torch.cat(all_y_pred))
    all_y = np.array(torch.cat(all_y))
    return all_y, all_y_pred


def compute_accuracy(vae, data_loader, classifier=None):
    all_y, all_y_pred = compute_predictions(vae, data_loader, classifier=classifier)
    return np.mean(all_y == all_y_pred)


def unsupervised_classification_accuracy(vae, data_loader, classifier=None):
    all_y, all_y_pred = compute_predictions(vae, data_loader, classifier=classifier)
    return unsupervised_clustering_accuracy(all_y, all_y_pred)


def compute_accuracy_nn(data_train, labels_train, data_test, labels_test, k=5):
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    return compute_accuracy_classifier(clf, data_train, labels_train, data_test, labels_test)


def compute_accuracy_classifier(clf, data_train, labels_train, data_test, labels_test):
    clf.fit(data_train, labels_train)
    # Predicting the labels
    y_pred_test = clf.predict(data_test)
    y_pred_train = clf.predict(data_train)

    return (compute_accuracy_tuple(labels_train, y_pred_train),
            compute_accuracy_tuple(labels_test, y_pred_test)), y_pred_test


def proximity_imputation(real_latent1, normed_gene_exp_1, real_latent2, k=4):
    knn = KNeighborsRegressor(k, weights='distance')
    y = knn.fit(real_latent1, normed_gene_exp_1).predict(real_latent2)
    return y
