use_cuda = True
import torch
from torch.utils.data import DataLoader
import numpy as np
from scvi.metrics.clustering import get_latent
from scvi.models import *
from scvi.inference import *
from scvi.harmonization.benchmark import sample_by_batch, knn_purity_avg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from scvi.dataset.BICCN import *
from scvi.harmonization.Seurat import SEURAT
from scvi.harmonization.Combat import COMBAT
from scvi.harmonization.benchmark import sample_by_batch, knn_purity_avg
from scvi.metrics.clustering import entropy_batch_mixing

# import sys
# load_model = bool(sys.argv[1])
# model_type = str(sys.argv[2])
load_model = False
model_type = 'svaec'
plotname = 'easycase2'



# np.savetxt("../Macosko_Regev.genenames.txt", gene_dataset.gene_names,fmt="%s")
# np.savetxt("../Macosko_Regev.label.txt", gene_dataset.labels, fmt="%s")
# np.save("../Macosko_Regev.batch.npy", np.concatenate(gene_dataset.batch_indices))
# mmwrite("../Macosko_Regev.count.npy", gene_dataset.X)




dataset1 = MacoskoDataset()
dataset2 = RegevDataset()

SEURAT = SEURAT()
seurat1 = SEURAT.create_seurat(dataset1,0)
seurat2 = SEURAT.create_seurat(dataset2,1)
ro.r.assign("seurat1", seurat1)
ro.r.assign("seurat2", seurat2)
combined = ro.r('hvg_CCA(seurat1,seurat2)')

combined = SEURAT.combine_seurat(dataset1,dataset2)
latent, batch_indices, labels, cell_types = SEURAT.get_cca(combined)
batch_entropy = entropy_batch_mixing(latent, batch_indices)
res = knn_purity_avg(latent, labels.astype('int'), cell_types, acc=True)

key_color_order = [['Pvalb low', 'Pvalb', 'Pvalb 1', 'Pvalb 2'],
                    ['Pvalb Ex_1','Pvalb Ex_2','Pvalb Ex'],
                    ['Pvalb Astro_1','Pvalb Astro_2'],
                    ['L2/3 IT Astro', 'L2/3 IT Macc1', 'L2/3 IT Sla_Astro', 'L2/3 IT', 'L2/3 IT Sla', 'L2/3 IT Sla_Inh'],
                    ['Sst Tac2', 'Sst Myh8', 'Sst Etv1', 'Sst Chodl', 'Sst'],
                    ['L5 PT_2', 'L5 PT IT',  'L5 PT_1'],
                    ['L5 IT Tcap_1_3', 'L5 IT Tcap_2', 'L5 IT Tcap_Astro', 'L5 IT Tcap_1', 'L5 IT Tcap_L2/3', 'L5 IT Tcap_Foxp2', 'L5 IT Tcap_3'],
                    ['L5 IT Aldh1a7_2','L5 IT Aldh1a7', 'L5 IT Aldh1a7_1'],
                    ['L5 NP', 'L5 NP Slc17a8'],
                    ['L6 IT Car3','L6 CT Olig','L6 IT Maf','L6 IT Ntn5 Mgp', 'L6 IT Ntn5 Inpp4b'],
                    ['L6 CT Nxph2',  'L6 CT Astro','L6 CT', 'L6 CT Grp'],
                    ['L6b', 'L6b F2r'],
                    ['Lamp5 Sncg', 'Lamp5 Egln3', 'Lamp5 Slc35d3'],
                    ['Vip Rspo4', 'Vip Serpinf1', 'Vip'],
                    ['Astro Ex', 'Astro Aqp4'],
                    ['OPC Pdgfra'],
                    ['VLMC Osr1'],
                    ['Oligo Enpp6_1','Oligo Enpp6_2','Oligo Opalin'],
                    ['Sncg Ptprk'],
                    ['Endo Slc38a5','Endo Slc38a5_Peri_2','Endo Slc38a5_Peri_1']]


key_color_order = [[key for key in key_color_group] for key_color_group in key_color_order]
ravel = [key for key_color in key_color_order for key in key_color]
labels_group_array = [len(key)*[group] for group, key in enumerate(key_color_order)]
labels_groups=[]
for group in labels_group_array:
    labels_groups+=group

dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
dataset.filter_cell_types(ravel)
dataset.subsample_genes(3000)
gene_dataset,labels_groups = combine_MacoskoRegev()

labels_groups, n = arrange_categories(np.array(labels_groups))


from scvi.dataset.cortex import CortexDataset
gene_dataset = CortexDataset()
gene_dataset.batch_indices = np.concatenate([np.repeat(0,1000),np.repeat(1,2005)])
from scvi.harmonization.Combat import COMBAT


if model_type is 'vae':
    if load_model is True:
        vae = torch.load('../easycase2.pt')
        data_loader = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda, shuffle=False,
                                 collate_fn=gene_dataset.collate_fn)
    else:
        vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
                  n_hidden=128, n_latent=10, n_layers=1, dispersion='gene')
        infer_vae = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
        infer_vae.train(n_epochs=250)
        torch.save(infer_vae.model, '../easycase2.pt')
        data_loader = infer_vae.data_loaders['sequential']
    latent, batch_indices, labels = get_latent(vae, data_loader)
    keys = gene_dataset.cell_types
    batch_indices = np.concatenate(batch_indices)
elif model_type is 'svaec':
    if load_model is True:
        svaec = torch.load('../easycase2.svaec.new_clust.pt')
        data_loader = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda, shuffle=False,
                                 collate_fn=gene_dataset.collate_fn)
    elif model_type is 'svaec':
        svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
                      gene_dataset.n_labels,use_labels_groups=True,labels_groups = list(labels_groups))
        # svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
        #               gene_dataset.n_labels,use_labels_groups=False)
        infer = JointSemiSupervisedVariationalInference(svaec, gene_dataset, n_labelled_samples_per_class=20)
        infer.train(n_epochs=50)
        infer.accuracy('unlabelled')
        torch.save(infer.model,'../easycase2.svaec.hierarchy.pt')
        data_loader = infer.data_loaders['unlabelled']
        latent, batch_indices, labels = get_latent(infer.model, infer.data_loaders['unlabelled'])
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)


n_plotcells = 6000
pop1 = 0
pop2 = 1
nbatches = len(np.unique(batch_indices))
_, cell_count = np.unique(batch_indices, return_counts=True)

sample = sample_by_batch(batch_indices, int(n_plotcells / nbatches))
sample_2batch = sample[(batch_indices[sample] == pop1) + (batch_indices[sample] == pop2)]
batch_entropy = entropy_batch_mixing(latent[sample_2batch, :], batch_indices[sample_2batch])
print("Entropy batch mixing :", batch_entropy)
compute_accuracy(svaec, data_loader, classifier=svaec.classifier)

sample = sample_by_batch(labels, 100)
latent_s = latent[sample, :]
batch_s = batch_indices[sample]
label_s = labels[sample]
if latent_s.shape[1] != 2:
    latent_s = TSNE().fit_transform(latent_s)


#
# res1 = knn_purity_avg(
#     latent[sample, :], labels[sample].astype('int'),
#     new_cell_types, acc=False
# )
# res2 = knn_purity_avg(
#     latent[sample, :], labels[sample].astype('int'),
#     new_cell_types, acc=True
# )
# print('average KNN purity')
# for x in res1:
#     print(x)
#
# print('average classification accuracy')
# for x in res2:
#     print(x)


colors = [sns.color_palette("Greens")[2:6], #Pvalb
sns.light_palette("green",5)[0:3] , #Pvalb Ex
sns.light_palette("green",5)[3:5], #Pvalb Astro
sns.light_palette("orange",6) , #L2/3
sns.light_palette('red')[1:6], #Sst
sns.light_palette("cyan",3) , # L5 PT
sns.light_palette('purple',8)[1:8] ,  # L5 IT Tcap
sns.light_palette('purple',7)[4:7] , # L5 IT Aldh1a7
sns.light_palette("navy",7)[3:5] , #L5 NP
sns.light_palette("brown",7)[2:7] , #L6 IT
sns.dark_palette("brown",8)[1:5] , #L6 CT
sns.dark_palette("green",8)[5:7] ,#L6
sns.dark_palette("yellow",7)[1:4] , #Lamp5
sns.dark_palette("yellow",7)[4:7] , #Vip
sns.color_palette("Paired",4), #Astro OPC VLMC
sns.color_palette('Greys',3), #Oligo
[sns.dark_palette('tan')], # sncg
sns.light_palette('hotpink',3)] # endo]

temp = []
for x in colors:
    temp = temp + x

colors = temp
key_color_order = ['Pvalb low', 'Pvalb', 'Pvalb 1', 'Pvalb 2',
                   'Pvalb Ex_1','Pvalb Ex_2','Pvalb Ex',
                   'Pvalb Astro_1','Pvalb Astro_2',
                    'L2/3 IT Astro', 'L2/3 IT Macc1', 'L2/3 IT Sla_Astro', 'L2/3 IT', 'L2/3 IT Sla', 'L2/3 IT Sla_Inh',
                    'Sst Tac2', 'Sst Myh8', 'Sst Etv1', 'Sst Chodl', 'Sst',
                    'L5 PT_2', 'L5 PT IT',  'L5 PT_1',
                    'L5 IT Tcap_1_3', 'L5 IT Tcap_2', 'L5 IT Tcap_Astro', 'L5 IT Tcap_1', 'L5 IT Tcap_L2/3', 'L5 IT Tcap_Foxp2', 'L5 IT Tcap_3',
                    'L5 IT Aldh1a7_2','L5 IT Aldh1a7', 'L5 IT Aldh1a7_1',
                    'L5 NP', 'L5 NP Slc17a8',
                    'L6 IT Car3','L6 CT Olig','L6 IT Maf','L6 IT Ntn5 Mgp', 'L6 IT Ntn5 Inpp4b',
                    'L6 CT Nxph2',  'L6 CT Astro','L6 CT', 'L6 CT Grp',
                    'L6b', 'L6b F2r',
                    'Lamp5 Sncg', 'Lamp5 Egln3', 'Lamp5 Slc35d3',
                    'Vip Rspo4', 'Vip Serpinf1', 'Vip',
                    'Astro Ex', 'Astro Aqp4',
                    'OPC Pdgfra',
                    'VLMC Osr1',
                    'Oligo Enpp6_1','Oligo Enpp6_2','Oligo Opalin',
                    'Sncg Ptprk',
                    'Endo Slc38a5','Endo Slc38a5_Peri_2','Endo Slc38a5_Peri_1']
clust_large = np.concatenate([np.repeat(0,4), #Pvalb
                              np.repeat(1,3), #Pvalb Ex
                              np.repeat(2,2), #Pvalb Astrol
                              np.repeat(3,6), #L2/3
                              np.repeat(4,5), #Sst
                              np.repeat(5,3), #L5 PT
                              np.repeat(6,7), # L5 IT Tcap_1_3
                              np.repeat(7,3), #L5 IT Aldh1a7_2
                              np.repeat(8,2), #L5 NP
                              np.repeat(9,5), #L6 IT
                              np.repeat(10,4), #L6 CT
                              np.repeat(11,2), #L6b
                              np.repeat(12,3), #Lamp5
                              np.repeat(13,3), #VIP
                              np.repeat(14,2), #Astro
                              np.repeat(15,1), #OPC
                              np.repeat(16,1), #VLMC
                              np.repeat(17,3), #oligo
                              np.repeat(18,1), # sncg
                              np.repeat(19,3) #Endo
])
clust_large_label =  np.concatenate([np.repeat('Pvalb',4), #Pvalb
                              np.repeat('Pvalb Ex',3), #Pvalb Ex
                              np.repeat('Pvalb Astrol',2), #Pvalb Astrol
                              np.repeat('L2/3',6), #L2/3
                              np.repeat('Sst',5), #Sst
                              np.repeat('L5 PT',3), #L5 PT
                              np.repeat('L5 IT Tcap',7), # L5 IT Tcap_1_3
                              np.repeat('L5 IT Aldh1a7',3), #L5 IT Aldh1a7_2
                              np.repeat('L5 NP',2), #L5 NP
                              np.repeat('L6 IT',5), #L6 IT
                              np.repeat('L6 CT',4), #L6 CT
                              np.repeat('L6b',2), #L6b
                              np.repeat('Lamp5',3), #Lamp5
                              np.repeat('VIP',3), #VIP
                              np.repeat('Astro',2), #Astro
                              np.repeat('OPC',1), #OPC
                              np.repeat('VLMC',1), #VLMC
                              np.repeat('Oligo',3), #oligo
                              np.repeat('Sncg',1), # sncg
                              np.repeat('Endo',3) #Endo
])

#
label_dict = dict(zip(keys,np.arange(len(keys))))
ordered_label = [label_dict[x] for x in key_color_order ]
label_dict = dict(zip(ordered_label,clust_large))
new_labels = np.asarray([label_dict[x] for x in labels])
new_cell_types = np.asarray(['Pvalb','Pvalb Ex','Pvalb Astrol','L2/3','Sst','L5PT','L5 IT Tcap','L5 IT Aldh1a7','L5 NP',
                             'L6 IT','L6 CT','L6b','Lamp5','VIP','Astro','OPC','VLMC','Oligo','Sncg','Endo'])

fig, ax = plt.subplots(figsize=(20, 20))
for i, k in enumerate(ordered_label):
    ax.scatter(latent_s[label_s == k, 0], latent_s[label_s == k, 1], c=colors[i], label=keys[k], edgecolors='none')

ax.legend()
fig.tight_layout()
fig.savefig('../' + plotname + '.' + model_type + '.label.png', dpi=fig.dpi)

plt.figure(figsize=(10, 10))
plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
plt.axis("off")
plt.tight_layout()
plt.savefig('../' + plotname + '.' + model_type + '.batch.png')

sample = sample_by_batch(new_labels, 50)
res1 = knn_purity_avg(
    latent[sample, :], new_labels[sample].astype('int'),
    new_cell_types, acc=False
)
res2 = knn_purity_avg(
    latent[sample, :], new_labels[sample].astype('int'),
    new_cell_types, acc=True
)
np.mean([x[1] for x in res2])
print('average KNN purity')
for x in res1:
    print(x)

print('average classification accuracy')
for x in res2:
    print(x)
