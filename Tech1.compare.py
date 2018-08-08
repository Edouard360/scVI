use_cuda = True
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.harmonization.utils_chenling import run_model, eval_latent
import sys
model_type = str(sys.argv[1])
plotname = 'Tech1'
from scvi.dataset.muris_tabula import TabulaMuris
dataset1 = TabulaMuris('facs')
dataset2 = TabulaMuris('droplet')
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
latent, batch_indices, labels, keys = run_model(model_type, gene_dataset, dataset1, dataset2, nlayers=3, filename=plotname, dispersion='gene-batch')
eval_latent(batch_indices, labels, latent, keys, plotname+'.'+model_type)
