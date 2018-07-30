from collections import namedtuple
from scvi.metrics.clustering import unsupervised_clustering_accuracy
import numpy as np
import torch

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


def compute_predictions(vae, data_loader, classifier=None):
    all_y_pred = []
    all_y = []

    for i_batch, tensors in enumerate(data_loader):
        sample_batch, _, _, _, labels = tensors
        all_y += [labels.view(-1)]

        if hasattr(vae, 'classify'):
            y_pred = vae.classify(sample_batch).argmax(dim=-1)
        elif classifier is not None:
            # Then we use the specified classifier
            if vae is not None:
                sample_batch, _, _ = vae.z_encoder(sample_batch)
            y_pred = classifier(sample_batch).argmax(dim=-1)
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
