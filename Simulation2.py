use_cuda = True
import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.conversion import ri2py
from scipy.sparse import csr_matrix

from scvi.dataset.dataset import GeneExpressionDataset
from scvi.harmonization.benchmark import knn_purity_avg
from scvi.inference import *
from scvi.models.scanvi import SCANVI
from scvi.models.vae import VAE

model_type = 'svaec'  # str(sys.argv[1])
plotname = 'EVFbatch_simulation'

countUMI = np.load('../sim_data/count.UMI.npy').T
countnonUMI = np.load('../sim_data/count.nonUMI.npy').T
labelUMI = np.load('../sim_data/label.UMI.npy')
labelnonUMI = np.load('../sim_data/label.nonUMI.npy')

UMI = GeneExpressionDataset(
            *GeneExpressionDataset.get_attributes_from_matrix(
                csr_matrix(countUMI), labels=labelUMI),
            gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])

nonUMI = GeneExpressionDataset(
            *GeneExpressionDataset.get_attributes_from_matrix(
                csr_matrix(countnonUMI), labels=labelnonUMI),
            gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])

if model_type in ['vae', 'svaec', 'Seurat', 'Combat']:
    gene_dataset = GeneExpressionDataset.concat_datasets(UMI, nonUMI)

    if model_type == 'vae':
        vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
                  n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
        infer_vae = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
        infer_vae.train(n_epochs=250)
        data_loader = infer_vae.data_loaders['sequential']
        latent, batch_indices, labels = get_latent(vae, data_loader)
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
        keys = gene_dataset.cell_types
    elif model_type == 'svaec':
        gene_dataset.subsample_genes(1000)

        n_epochs_vae = 100
        n_epochs_scanvi = 50
        vae = VAE(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels, n_latent=10, n_layers=2)
        trainer = UnsupervisedTrainer(vae, gene_dataset, train_size=1.0)
        trainer.train(n_epochs=n_epochs_vae)

        for i in [0, 1]:
            scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels, n_layers=2)
            scanvi.load_state_dict(vae.state_dict(), strict=False)
            trainer_scanvi = SemiSupervisedTrainer(scanvi, gene_dataset, classification_ratio=1,
                                                   n_epochs_classifier=1, lr_classification=5 * 1e-3)

            trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(indices=(gene_dataset.batch_indices == i))
            trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(
                indices=(gene_dataset.batch_indices == 1 - i)
            )

            trainer_scanvi.model.eval()
            print('NN: ', trainer_scanvi.nn_latentspace())
            trainer_scanvi.unlabelled_set.to_monitor = ['accuracy']
            trainer_scanvi.labelled_set.to_monitor = ['accuracy']
            trainer_scanvi.full_dataset.to_monitor = ['entropy_batch_mixing']
            trainer_scanvi.train(n_epochs=n_epochs_scanvi)

            if i == 0:
                print("Score UMI->nonUMI:", trainer_scanvi.unlabelled_set.accuracy())
            else:
                print("Score nonUMI->UMI:", trainer_scanvi.unlabelled_set.accuracy())
    elif model_type == 'Seurat':
        from scvi.harmonization.clustering.seurat import SEURAT
        SEURAT = SEURAT()
        seurat1 = SEURAT.create_seurat(UMI, 0)
        seurat2 = SEURAT.create_seurat(nonUMI, 1)
        latent, batch_indices,labels = SEURAT.combine_seurat(seurat1, seurat2)
        numpy2ri.activate()
        latent  = ri2py(latent)
        batch_indices  = ri2py(batch_indices)
        labels  = ri2py(labels)
        keys,labels = np.unique(labels,return_inverse=True)
        latent  = np.array(latent)
        batch_indices  = np.array(batch_indices)
        labels = np.array(labels)
    elif model_type == 'Combat':
        from scvi.harmonization.clustering.combat import COMBAT
        COMBAT = COMBAT()
    # corrected = COMBAT.combat_correct(gene_dataset)
        latent = COMBAT.combat_pca(gene_dataset)
        latent = latent.T
        batch_indices = np.concatenate(gene_dataset.batch_indices)
        labels = np.concatenate(gene_dataset.labels)
        keys = gene_dataset.cell_types

    sample = select_indices_evenly(2000, batch_indices)
    batch_entropy = entropy_batch_mixing(latent[sample, :], batch_indices[sample])
    print("Entropy batch mixing :", batch_entropy)

    if model_type == 'svaec':
        svaec_acc = compute_accuracy(svaec, data_loader, classifier=svaec.classifier)

    sample = select_indices_evenly(1000, labels)
    res = knn_purity_avg(
        latent[sample, :], labels[sample].astype('int'),
        keys=keys, acc=True
    )

    print('average classification accuracy per cluster')
    for x in res:
        print(x)

    knn_acc = np.mean([x[1] for x in res])
    print("average KNN accuracy:", knn_acc)

    sample = select_indices_evenly(1000, labels)
    latent_s = latent[sample, :]
    batch_s = batch_indices[sample]
    label_s = labels[sample]
    if latent_s.shape[1] != 2:
        latent_s = TSNE().fit_transform(latent_s)

    colors = sns.color_palette('tab10', 5)
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, k in enumerate(keys):
        ax.scatter(latent_s[label_s == i, 0], latent_s[label_s == i, 1], c=colors[i], label=k, edgecolors='none')

    ax.legend()
    fig.tight_layout()
    fig.savefig('../' + plotname + '.' + model_type + '.label.png', dpi=fig.dpi)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('../' + plotname + '.' + model_type + '.batch.png')
else:
    from scvi.harmonization import SCMAP

    scmap = SCMAP()
    with open('Simulation2-scmap.txt', 'w') as file:
        print("Starting scmap")
        for n_features in [100, 300, 500, 1000, 2000]:
            scmap.set_parameters(n_features=n_features)
            scmap.fit_scmap_cluster(countUMI, labelUMI.astype(np.int))
            line = "Score UMI->nonUMI:%.4f  [n_features = %d]\n" % (scmap.score(countnonUMI, labelnonUMI), n_features)
            file.write(line)

            scmap.fit_scmap_cluster(countnonUMI, labelnonUMI.astype(np.int))
            line = "Score nonUMI->UMI:%.4f  [n_features = %d]\n" % (scmap.score(countUMI, labelUMI), n_features)
            file.write(line)
