import numpy as np


def get_statistics(res_a, res_b, M_p=10000, permutation=False):
    """
    Output average over statistics in a symmetric way (a against b)
    forget the sets if permutation is True
    """

    # agregate dataset
    samples = np.vstack((res_a["sample_rate"], res_b["sample_rate"]))

    # prepare the pairs for sampling
    list_1 = list(np.arange(res_a["sample_rate"].shape[0]))
    list_2 = list(res_a["sample_rate"].shape[0] + np.arange(res_b["sample_rate"].shape[0]))
    if not permutation:
        # case1: no permutation, sample from A and then from B
        u, v = np.random.choice(list_1, size=M_p), np.random.choice(list_2, size=M_p)
    else:
        # case2: permutation, sample from A+B twice
        u, v = (np.random.choice(list_1 + list_2, size=M_p), \
                np.random.choice(list_1 + list_2, size=M_p))

    # then constitutes the pairs
    first_set = samples[u]
    second_set = samples[v]

    res = np.mean(first_set >= second_set, 0)
    res = np.log(res) - np.log(1 - res)  # +1e-8+1e-8
    return res


def sample_posterior(model, X, M_z, trainer_ref=None, sess=None, expression=None, training_phase=None, kl_scalar=None):
    # shape and simulation
    results = {}
    ind = np.arange(X.shape[0])

    # We know for sure the labels here.

    # repeat the data for sampling
    X_m = np.repeat(X, M_z, axis=0)
    ind = np.repeat(ind, M_z, axis=0)

    if trainer_ref is not None:
        from scvi.dataset import GeneExpressionDataset
        data_subset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(X_m))
        tmp_posterior = trainer_ref.create_posterior(model, data_subset)
        print(tmp_posterior.data_loader_kwargs)
        scale = tmp_posterior.get_sample_scale()
    else:
        import tensorflow as tf
        dic_x = {expression: X_m, training_phase: False, kl_scalar: 1.}
        z_m, l_m = sess.run((model.z, model.library), feed_dict=dic_x)
        dic_z = {model.z: z_m, model.library: l_m, training_phase: False, kl_scalar: 1.}
        rate, dropout, scale = sess.run((model.px_rate, model.px_dropout, model.px_scale), feed_dict=dic_z)
        dispersion = np.tile(sess.run((tf.exp(model.px_r))), (rate.shape[0], 1))

    results["sample_rate"] = scale
    return results

def get_sampling(model, subset_a, subset_b, M_z, trainer_ref=None, expression_train=None, sess=None, expression=None, training_phase=None, kl_scalar=None):
    #get q(z| xa) and q(z| xb) and sample M times from it, then output gamma parametrizations
    res_a = sample_posterior(model, expression_train[subset_a], M_z,trainer_ref, sess=sess, expression=expression, training_phase=training_phase, kl_scalar=kl_scalar)
    res_b = sample_posterior(model, expression_train[subset_b], M_z, trainer_ref, sess=sess, expression=expression, training_phase=training_phase, kl_scalar=kl_scalar)
    return res_a, res_b
