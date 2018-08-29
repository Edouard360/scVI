'''
This file only concerns the romain's code for the tensorflow model.
'''
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import scipy.stats as stats
import time
import tensorflow as tf
import pandas as pd
from main_debug_statistics import get_sampling, get_statistics
from romain_model import scVIModel as scVI
import matplotlib.pyplot as plt
from romain_helper import eval_params, train_model, eval_library, eval_qz_v


data_path = "/home/ubuntu/10xPBMCs/"
expression_train = np.load(data_path + "de/data_train.npy")
c_train = np.loadtxt(data_path + "label_train")
learning_rate = 0.0004
epsilon = 0.01


print(expression_train.shape)
# batch_size, nb_genes = 100,50
# data = np.random.negative_binomial(5, 0.3, size=(batch_size, nb_genes))
# mask = np.random.binomial(n=1, p=0.7, size=(batch_size, nb_genes))
# expression_train = (data * mask)




# getting priors
log_library_size = np.log(np.sum(expression_train, axis=1))
mean, var = np.mean(log_library_size), np.var(log_library_size)

scores=[]
n_epochs=100
for _ in range(15):
    tf.reset_default_graph()

    expression = tf.placeholder(tf.float32, (None, expression_train.shape[1]), name='x')
    kl_scalar = tf.placeholder(tf.float32, (), name='kl_scalar')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    training_phase = tf.placeholder(tf.bool, (), name='training_phase')

    model = scVI(expression=expression, kl_scale=kl_scalar, \
                              optimize_algo=optimizer, phase=training_phase, \
                               library_size_mean=mean, library_size_var=var, n_latent=10)

    # Session creation

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    result = train_model(model, (expression_train, None), sess, n_epochs)

    interest = "BDC"
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
    res_a, res_b = get_sampling(model, subset_a, subset_b, M_z, None, expression_train, sess=sess, expression=expression, training_phase=training_phase, kl_scalar=kl_scalar)
    st = get_statistics(res_a, res_b, M_p=40000)

    print("Detected ", np.sum(2 * np.abs(st) >= 6), " genes with scVI")

    scores+=[np.sum(2 * np.abs(st) >= 6)]

    rate, dispersion, dropout = eval_params(model, expression_train, sess)

    variance_nb = ((dispersion+rate)/dispersion)*rate

    # st_permutation = get_statistics(res_a, res_b, M_p=40000, permutation=True)
    # st_permutation.var()


    # variance_nb.mean(axis=0).max() # 516781.7
    # rate.mean(axis=0).max() # 387.01816




    # library = eval_library(model, expression_train, sess)

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

    # qz_v = eval_qz_v(model, expression_train, sess)
    # print(np.sqrt(qz_v.mean()))
    #
    # print(variance_nb.mean(axis=0).max()  )# 4776.633 ( against: 516781.7 )
    # print(variance_nb.mean(axis=0).mean()  )# 5.80 (against: 293.25)
    # print(rate.mean(axis=0).max()  )# 46.19 ( against : 387.01816 )
    # print(rate.mean(axis=0).mean() )# 0.389 ( against: 0.623)
    # print(dispersion.mean(axis=0).max())
    # print(dispersion.mean(axis=0).mean())


# In [1]: scores
# Out[1]: [1173, 1084, 1051, 1143, 958]

# Add dropout
# Set seed to 0


# In [1]: scores
# Out[1]:
# [1667,
#  1773,
#  1647,
#  1536,
#  1779,
#  1552,
#  1645,
#  1768,
#  1613,
#  1593,
#  1457,
#  1656,
#  1539,
#  1660,
#  1662]
