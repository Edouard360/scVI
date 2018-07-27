from scvi.dataset.BICCN import MacoskoDataset,RegevDataset
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.models.vae import VAE
from scvi.inference import VariationalInference
use_cuda = True
import torch
from torch.utils.data import DataLoader
import numpy as np
from scvi.metrics.clustering import get_latent
from scvi.harmonization.benchmark import sample_by_batch, knn_purity_avg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
plotname = 'easycase2'
from scvi.metrics.clustering import entropy_batch_mixing

dataset1 = MacoskoDataset()
dataset2 = RegevDataset()

temp1=[]
for key in label_map2.keys():
    temp1.append(label_map2[key].split('_')[0])

temp2=[]
for key in label_map1.keys():
    temp2.append(label_map1[key].split('_')[0])

gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
gene_dataset.subsample_genes(5000)

load_model = True

if load_model is True:
    vae = torch.load('../easycase2.pt')
else:
    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
              n_hidden=128, n_latent=10, n_layers=1, dispersion='gene')
    infer_vae = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
    infer_vae.fit(n_epochs=250)

data_loader = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda, shuffle=False,
                         collate_fn=gene_dataset.collate_fn)
latent, batch_indices, labels = get_latent(vae, data_loader)
keys = gene_dataset.cell_types
batch_indices = np.concatenate(batch_indices)

# def EvaluateHarmonization(pop1,pop2,plotcells):
n_plotcells = 6000
pop1 = 0
pop2 = 1
nbatches = len(np.unique(batch_indices))
_, cell_count = np.unique(batch_indices, return_counts=True)

sample = sample_by_batch(batch_indices, int(n_plotcells / nbatches))
sample_2batch = sample[(batch_indices[sample] == pop1) + (batch_indices[sample] == pop2)]
batch_entropy = entropy_batch_mixing(latent[sample_2batch, :], batch_indices[sample_2batch])
print("Entropy batch mixing :", batch_entropy)

latent_s = latent[sample, :]
batch_s = batch_indices[sample]
label_s = labels[sample]
if latent_s.shape[1] != 2:
    latent_s = TSNE().fit_transform(latent_s)

plt.figure(figsize=(10, 10))
plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
plt.axis("off")
plt.tight_layout()
plt.savefig('../' + plotname + '.batch.png')


sample = sample_by_batch(labels, 100)
latent_s = latent[sample, :]
batch_s = batch_indices[sample]
label_s = labels[sample]
if latent_s.shape[1] != 2:
    latent_s = TSNE().fit_transform(latent_s)

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

label_dict = dict(zip(keys,np.arange(len(keys))))
ordered_label = [label_dict[x] for x in key_color_order ]


fig, ax = plt.subplots(figsize=(20, 20))
for i, k in enumerate(ordered_label):
    ax.scatter(latent_s[label_s == k, 0], latent_s[label_s == k, 1], c=colors[i], label=keys[k], edgecolors='none')

ax.legend()
fig.tight_layout()
fig.savefig('../' + plotname + '.label.png', dpi=fig.dpi)


sample_2batch = np.random.permutation(sample_2batch)[:1000]

label_map1 = np.genfromtxt('../AIBS/' + '10X_nuclei_Macosko/cluster.annotation.csv', dtype='str', delimiter=',')
label_map2 = np.genfromtxt('../AIBS/' + '10X_nuclei_Regev/cluster.annotation.csv', dtype='str', delimiter=',')
label_map1 = dict(zip([x.split('"')[1] for x in label_map[:, 0]], [x.split('"')[1] for x in label_map1[:, 1]]))
label_map2 = dict(zip([x.split('"')[1] for x in label_map[:, 0]], [x.split('"')[1] for x in label_map2[:, 1]]))

clust_large = np.concatenate([np.repeat(0,3), #Pvalb
                              np.repeat(1,3), #Pvalb Ex
                              np.repeat(2,2), #Pvalb Astrol
                              np.repeat(3,6), #L2/3
                              np.repeat(4,5), #Sst
                              np.repeat(5,3), #L5 PT
                              np.repeat(6,7), # L5 IT Tcap_1_3
                              np.repeat(7,3), #L5 IT Aldh1a7_2
                              np.repeat(8,5), #L6 IT
                              np.repeat(9,1),
                              np.repeat(10,1),
                              np.repeat(11,1),
                              np.repeat(12,1),
                              np.repeat(13,1)
])
res1 = knn_purity_avg(
    latent[sample_2batch, :], labels[sample_2batch].astype('int'),
    gene_dataset.cell_types, acc=False
)
res2 = knn_purity_avg(
    latent[sample_2batch, :], labels[sample_2batch].astype('int'),
    gene_dataset.cell_types, acc=True
)
print('average KNN purity')
for x in res1:
    print(x)

print('average classification accuracy')
for x in res2:
    print(x)


