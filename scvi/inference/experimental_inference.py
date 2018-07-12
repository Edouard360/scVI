import torch

from scvi.inference import JointSemiSupervisedVariationalInference, VariationalInference
from scvi.models import InfoCatVAEC


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
