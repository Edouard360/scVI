'''
This file requires the environment to contain both pytorch and tf.
It instanciates pytorch's weights with that of tensorflow.
Either use Synthetic data (to debug) or PBMC.
Evaluate, with the same metrics the DE for Dendritic / B_cells
'''
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import scipy.stats as stats
import time
import tensorflow as tf
import pandas as pd
import torch
from scvi.dataset import GeneExpressionDataset
from main_debug_statistics import get_sampling, get_statistics
from romain_model import scVIModel as scVI
import matplotlib.pyplot as plt
from romain_helper import eval_params, train_model, eval_library, eval_qz_v
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer

#data_path = "/home/ubuntu/10xPBMCs/"
#expression_train = np.load(data_path + "de/data_train.npy")
#c_train = np.loadtxt(data_path + "label_train")
learning_rate = 0.0004
epsilon = 0.01
np.random.seed(1)
batch_size, nb_genes = 200,50
data = np.random.negative_binomial(5, 0.3, size=(batch_size, nb_genes))
mask = np.random.binomial(n=1, p=0.7, size=(batch_size, nb_genes))
expression_train = (data * mask)

# TODO : TF
tf.reset_default_graph()
tf.set_random_seed(0)
expression = tf.placeholder(tf.float32, (None, expression_train.shape[1]), name='x')
kl_scalar = tf.placeholder(tf.float32, (), name='kl_scalar')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)#, #epsilon=epsilon)
training_phase = tf.placeholder(tf.bool, (), name='training_phase')

# getting priors
log_library_size = np.log(np.sum(expression_train, axis=1))
mean, var = np.mean(log_library_size), np.var(log_library_size)

model = scVI(expression=expression, kl_scale=kl_scalar, \
                          optimize_algo=optimizer, phase=training_phase, \
                           library_size_mean=mean, library_size_var=var, n_latent=10, dropout_rate=0)

# Session creation
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
np.random.seed(0)
result = train_model(model, (expression_train, None), sess, 1)


# ====== Other arguments to compute
list_parameters_names_tf = [v.name for v in tf.trainable_variables()]
#print(list_parameters_names_tf)
# print(len(list_parameters_names_tf))
#same_input = [v for v in tf.trainable_variables() if v.name == 'variational_distribution/fully_connected/weights:0'][0]

tf.set_random_seed(0)
# TODO : PYTORCH
from itertools import cycle
z_tensor = tf.random_normal([128,10], 0.0, 1.0, seed=0)
l_tensor = tf.random_normal([128, 1], 0.0, 1.0, seed=0)
with tf.Session() as sess1:
    samples_z = [sess1.run(z_tensor) for i in range(10)]
    samples_l = [sess1.run(l_tensor) for i in range(10)]


pbmc_dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(expression_train))
pbmc_vae = VAE(pbmc_dataset.nb_genes)

pbmc_vae.z_encoder.iter_samples = cycle(iter(samples_z))
pbmc_vae.l_encoder.iter_samples = cycle(iter(samples_l))

pbmc_trainer = UnsupervisedTrainer(pbmc_vae,
                                   pbmc_dataset,
                                   train_size=1.0,
                                   verbose=True,
                                   frequency=10)
pbmc_trainer.train_set.to_monitor = ['ll','get_stats']
pbmc_trainer.test_set.to_monitor = []

#same_input_np = tf.convert_to_tensor(tf.get_variable('variational_distribution/fully_connected/weights:0'), np.float32)
#same_input_np = tf.convert_to_tensor(same_input, np.float32)




pytorch_parameters_dict = dict(pbmc_vae.named_parameters())
list_parameters_names_pytorch = [p[0] for p in pbmc_vae.named_parameters()]
print(len(list_parameters_names_pytorch))
interest = "BDC"
couple_celltypes = (4, 0)
rank_auc = 800
p_prior = 0.25

list_parameters_names_pytorch = list_parameters_names_pytorch[1:]+[list_parameters_names_pytorch[0]]
print(list_parameters_names_tf)
for p1, p2 in zip(list_parameters_names_pytorch, list_parameters_names_tf):


    #p2_tensor = tf.convert_to_tensor(tf.get_variable(p2), np.float32)
    #p2_tensor_np = sess.run(p2_tensor)
    # print(p2)
    p2_numpy = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, p2))[0]
    # print(p2_numpy.mean())
    # print(p2_numpy.var())
    # print(p2_numpy.shape)
    data = pytorch_parameters_dict[p1].data
    pytorch_parameters_dict[p1].data = torch.from_numpy(p2_numpy.T).to(device = data.device).type(data.dtype)
    # print(p1)
    # p1_numpy = pytorch_parameters_dict[p1].data.numpy().T
    # print(p1_numpy.shape)
    # print(p1_numpy.mean())
    # print(p1_numpy.var())

print()
np.random.seed(0)
pbmc_trainer.train(n_epochs=1, lr=learning_rate, eps=epsilon)

# getting p_values

# setting up parameters
A, B, M_z = 300, 300, 200
# set_a = np.where(c_train == couple_celltypes[0])[0]
# set_b = np.where(c_train == couple_celltypes[1])[0]

# subsampling cells and computing statistics
subset_a = np.random.choice(set_a, A)
subset_b = np.random.choice(set_b, B)
res_a, res_b = get_sampling(model, subset_a, subset_b, M_z, None, expression_train, sess=sess, expression=expression, training_phase=training_phase, kl_scalar=kl_scalar)
st = get_statistics(res_a, res_b, M_p=40000)

print("Detected ", np.sum(2 * np.abs(st) >= 6), " genes with scVI")


rate, dispersion, dropout = eval_params(model, expression_train, sess)

variance_nb = ((dispersion+rate)/dispersion)*rate

st_permutation = get_statistics(res_a, res_b, M_p=40000, permutation=True)
st_permutation.var()


variance_nb.mean(axis=0).max() # 516781.7
rate.mean(axis=0).max() # 387.01816




library = eval_library(model, expression_train, sess)
# TENSORFLOW

# 137000.69
# 74.9298
# 178.8039
# 0.44984102
# 34.478813
# 2.024544

# 366089.25
# 138.15508
# 178.52783
# 0.5418974
# 25.621471
# 2.0333593

# PYTORCH

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

qz_v = eval_qz_v(model, expression_train, sess)
print(np.sqrt(qz_v.mean()))

print(variance_nb.mean(axis=0).max()  )# 4776.633 ( against: 516781.7 )
print(variance_nb.mean(axis=0).mean()  )# 5.80 (against: 293.25)
print(rate.mean(axis=0).max()  )# 46.19 ( against : 387.01816 )
print(rate.mean(axis=0).mean() )# 0.389 ( against: 0.623)
print(dispersion.mean(axis=0).max())
print(dispersion.mean(axis=0).mean())
