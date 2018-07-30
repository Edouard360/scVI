import numpy as np

from . import GeneExpressionDataset


class SyntheticDataset(GeneExpressionDataset):
    def __init__(self, batch_size=200, nb_genes=100, n_batches=2, n_labels=3):
        # Generating samples according to a ZINB process
        data = np.random.negative_binomial(5, 0.3, size=(n_batches, batch_size, nb_genes))
        mask = np.random.binomial(n=1, p=0.7, size=(n_batches, batch_size, nb_genes))
        newdata = (data * mask)  # We put the batch index first
        labels = np.random.randint(0, n_labels, size=(n_batches, batch_size, 1))
        super(SyntheticDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list(newdata, list_labels=labels),
            gene_names=np.arange(nb_genes).astype(np.str), cell_types=np.arange(n_labels).astype(np.str)
        )


class SyntheticSimilar(GeneExpressionDataset):
    def __init__(self):
        cluster_1 = np.load("data/sim_batch/label1.npy").astype(np.int)
        count_1 = np.load("data/sim_batch/count1.npy")
        cluster_1 = cluster_1 - 1

        cluster_2 = np.load("data/sim_batch/label2.npy").astype(np.int)
        count_2 = np.load("data/sim_batch/count2.npy")
        cluster_2 = cluster_2 - 1

        super(SyntheticSimilar, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list([count_1, count_2], list_labels=[cluster_1, cluster_2])
        )


class SyntheticDifferent(GeneExpressionDataset):
    def __init__(self):
        cluster_1 = np.load("data/sim_batch/label.nonUMI.npy").astype(np.int)
        count_1 = np.load("data/sim_batch/count.nonUMI.npy")
        cluster_1 = cluster_1 - 1

        cluster_2 = np.load("data/sim_batch/label.UMI.npy").astype(np.int)
        count_2 = np.load("data/sim_batch/count.UMI.npy")
        cluster_2 = cluster_2 - 1

        super(SyntheticDifferent, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list([count_1, count_2], list_labels=[cluster_1, cluster_2])
        )
