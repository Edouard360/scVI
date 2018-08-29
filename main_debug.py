'''
This file only concerns the pytorch code. It has worked well on DE for the current versionning parameters
'''

from scvi.dataset import GeneExpressionDataset, SyntheticDataset
from scvi.inference import *
from scvi.models import *
import numpy as np

from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer

n_epochs=100
lr=0.0004
use_batches=False
use_cuda=True

# train the model

from main_debug_statistics import get_statistics, get_sampling, sample_posterior

data_path = "/home/ubuntu/10xPBMCs/"
expression_train = np.load(data_path + "de/data_train.npy")
c_train = np.loadtxt(data_path + "label_train")

pbmc_dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(expression_train))
#pbmc_dataset = SyntheticDataset()

pbmc_vae = VAE(pbmc_dataset.nb_genes)
pbmc_trainer = UnsupervisedTrainer(pbmc_vae,
                                   pbmc_dataset,
                                   train_size=1.0,
                                   verbose=True,
                                   frequency=10,
                                   use_cuda=use_cuda)
pbmc_trainer.train_set.to_monitor = ['ll','get_stats']
pbmc_trainer.test_set.to_monitor = []
pbmc_trainer.train(n_epochs=n_epochs, lr=lr, eps=0.01)


interest = "BDC"
print(interest)
couple_celltypes = (4, 0)
rank_auc = 800
p_prior = 0.25


# setting up parameters
A, B, M_z = 300, 300, 200
set_a = np.where(c_train == couple_celltypes[0])[0]
set_b = np.where(c_train == couple_celltypes[1])[0]

# subsampling cells and computing statistics
subset_a = np.random.choice(set_a, A)
subset_b = np.random.choice(set_b, B)
res_a, res_b = get_sampling(pbmc_vae, subset_a, subset_b, M_z, pbmc_trainer, expression_train)

st = get_statistics(res_a, res_b, M_p=40000)
print("Detected ", np.sum(2 * np.abs(st) >= 6), " genes with scVI")

st_permutation = get_statistics(res_a, res_b, M_p=40000, permutation=True)
print("Detected ", np.sum(2 * np.abs(st) >= 6), " genes with scVI")


dropout, rate, dispersion = pbmc_trainer.train_set.generate_parameters()

variance_nb = ((dispersion+rate)/dispersion)*rate

library = pbmc_trainer.train_set.get_library()

print(variance_nb.mean(axis=0).max()  )# 4776.633 ( against: 516781.7 )
print(variance_nb.mean(axis=0).mean()  )# 5.80 (against: 293.25)
print(rate.mean(axis=0).max()  )# 46.19 ( against : 387.01816 )
print(rate.mean(axis=0).mean() )# 0.389 ( against: 0.623)
print(dispersion.mean(axis=0).max())
print(dispersion.mean(axis=0).mean())

# 9869.554
# 6.6310263
# 44.97964
# 0.42594424
# 30.222311
# 2.0557868

# 5384.078
# 7.7088757
# 47.94623
# 0.4606595
# 33.684135
# 2.1179774

