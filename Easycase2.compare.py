from scvi.harmonization.utils_chenling import run_model, eval_latent
use_cuda = True
from scvi.dataset.BICCN import *
from scvi.dataset.dataset import GeneExpressionDataset

import sys
model_type = str(sys.argv[1])
plotname = 'Macosko_Regev'

if model_type!='readSeurat':
    dataset1 = MacoskoDataset()
    dataset2 = RegevDataset()
    gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
    gene_dataset.subsample_genes(5000)

    if model_type=='writedata':
        latent, batch_indices, labels, keys = run_model(model_type, gene_dataset, dataset1, dataset2, plotname)
    else:
        latent, batch_indices, labels, keys = run_model(model_type, gene_dataset, dataset1, dataset2)
elif model_type=='readSeurat':
    latent, batch_indices, labels, keys = run_model(model_type, 0, 0, 0, plotname)

groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
          'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
          'SMC', 'MICRO']
groups = np.asarray([x.upper() for x in groups])
cell_type_bygroup = np.concatenate([[x for x in keys if x.startswith(y)] for y in groups])
new_labels_dict = dict(zip(cell_type_bygroup, np.arange(len(cell_type_bygroup))))
labels = np.asarray([keys[x] for x in labels])
new_labels = np.asarray([new_labels_dict[x] for x in labels])
labels_groups = [[i for i, x in enumerate(groups) if y.startswith(x)][0] for y in cell_type_bygroup]
coarse_labels_dict = dict(zip(np.arange(len(labels_groups)), labels_groups))
coarse_labels = np.asarray([coarse_labels_dict[x] for x in new_labels]).astype('int')
groups = groups[np.unique(coarse_labels)]
mapping = dict(zip(np.unique(coarse_labels),np.arange(len(np.unique(coarse_labels)))))
coarse_labels = np.asarray([mapping[x] for x in coarse_labels])
eval_latent(batch_indices, coarse_labels, latent, groups, plotname+'.'+model_type)
