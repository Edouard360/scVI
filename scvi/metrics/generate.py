import torch

from scvi.utils import to_cuda, no_grad, eval_modules


@no_grad()
@eval_modules()
def generate(vae, data_loader):
    generated = []
    all_labels = []
    for tensorlist in data_loader:
        if vae.use_cuda:
            tensorlist = to_cuda(tensorlist)
        sample_batch, local_l_mean, _, _, _ = tensorlist
        g, labels = vae.generate(sample_batch, local_l_mean)

        generated += [g]
        all_labels += [labels]
    return torch.cat(generated), torch.cat(all_labels)
