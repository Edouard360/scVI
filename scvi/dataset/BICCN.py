from scvi.harmonization.utils_chenling import get_matrix_from_dir,get_matrix_from_h5,TryFindCells
import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import os.path

class MacoskoDataset(GeneExpressionDataset):
    def __init__(self, save_path='../AIBS/'):
        self.save_path = save_path
        count, labels, cell_type, gene_names = self.preprocess()
        super(MacoskoDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)

    def preprocess(self):
        if os.path.isfile(self.save_path + 'macosko_data.svmlight'):
            count, labels = load_svmlight_file(self.save_path + 'macosko_data.svmlight')
            cell_type = np.load(self.save_path + 'macosko_data.celltypes.npy')
            gene_names = np.load(self.save_path + 'macosko_data.gene_names.npy')
            return(count,labels,cell_type,gene_names)
        else:
            macosko_batches = ['a','b','c','d','e','f','g','h']
            label = np.genfromtxt(self.save_path + '10X_nuclei_Macosko/cluster.membership.csv', dtype='str', delimiter=',')
            label_batch = np.asarray([str(int(int(x.split('-')[1].split('"')[0])/11)) for x in label[1:,0]])
            label_barcode = np.asarray([x.split('-')[0].split('"')[1] for x in label[1:,0]])
            label_cluster = np.asarray([x.split('"')[1] for x in label[1:,1]])
            label_map = np.genfromtxt(self.save_path + '10X_nuclei_Macosko/cluster.annotation.csv', dtype='str', delimiter=',')
            label_map = dict(zip([x.split('"')[1] for x in label_map[:, 0]], [x.split('"')[1] for x in label_map[:, 1]]))
            macosko_data = []
            for batch_i, batch in enumerate(macosko_batches):
                count, geneid, cellid = get_matrix_from_dir(self.save_path + '10X_nuclei_Macosko/'+'171218_p56m1'+ batch)
                geneid = geneid[:,1]
                count = count.T.tocsr()
                print(count.shape,len(geneid),len(cellid))
                cellid = [id.split('-')[0] for id in cellid]
                label_dict = dict(zip(label_barcode[label_batch == str(batch_i+1)],label_cluster[label_batch == str(batch_i+1)]))
                new_count, matched_label = TryFindCells(label_dict, cellid, count)
                new_label = np.repeat(0,len(matched_label))
                for i,x in enumerate(np.unique(matched_label)):
                    new_label[matched_label == x] = i
                cell_type = [label_map[x] for x in np.unique(matched_label)]
                dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(new_count, labels=new_label),
                                            gene_names=geneid, cell_types=cell_type)
                print(dataset.X.shape,len(dataset.labels))
                if len(macosko_data)>0:
                    macosko_data = GeneExpressionDataset.concat_datasets(macosko_data,dataset)
                else:
                    macosko_data = dataset
            dump_svmlight_file(macosko_data.X,np.concatenate(macosko_data.labels),self.save_path +'macosko_data.svmlight')
            np.save(self.save_path + 'macosko_data.celltypes.npy',macosko_data.cell_types)
            np.save(self.save_path + 'macosko_data.gene_names.npy',macosko_data.gene_names)
            count, labels = load_svmlight_file(self.save_path + 'macosko_data.svmlight')
            cell_type = np.load(self.save_path + 'macosko_data.celltypes.npy')
            gene_names = np.load(self.save_path + 'macosko_data.gene_names.npy')
            return(count,labels,cell_type,gene_names)



class RegevDataset(GeneExpressionDataset):
    def __init__(self, save_path='../AIBS/'):
        self.save_path = save_path
        count,labels,cell_type,gene_names = self.preprocess()
        super(RegevDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)

    def preprocess(self):
        if os.path.isfile(self.save_path + 'regev_data.svmlight'):
            count, labels = load_svmlight_file(self.save_path + 'regev_data.svmlight')
            cell_type = np.load(self.save_path + 'regev_data.celltypes.npy')
            gene_names = np.load(self.save_path + 'regev_data.gene_names.npy')
            return(count,labels,cell_type,gene_names)
        else:
            regev_batches = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            label = np.genfromtxt(self.save_path + '10X_nuclei_Regev/cluster.membership.csv', dtype='str', delimiter=',')
            label_batch = np.asarray([str(int(int(x.split('-')[1].split('"')[0]))) for x in label[1:, 0]])
            label_barcode = np.asarray([x.split('-')[0].split('"')[1] for x in label[1:, 0]])
            label_cluster = np.asarray([x.split('"')[1] for x in label[1:, 1]])
            label_map = np.genfromtxt(self.save_path + '10X_nuclei_Regev/cluster.annotation.csv', dtype='str', delimiter=',')
            label_map = dict(zip([x.split('"')[1] for x in label_map[:, 0]], [x.split('"')[1] for x in label_map[:, 1]]))
            regev_data = []
            for batch_i, batch in enumerate(regev_batches):
                _, geneid, cellid, count = get_matrix_from_h5(
                    self.save_path + '10X_nuclei_Regev/' + batch + '1/filtered_gene_bc_matrices_h5.h5', 'mm10-1.2.0_premrna')
                count = count.T.tocsr()
                cellid = [id.split('-')[0] for id in cellid]
                print(count.shape, len(geneid), len(cellid))
                label_dict = dict(
                    zip(label_barcode[label_batch == str(batch_i + 1)], label_cluster[label_batch == str(batch_i + 1)]))
                new_count, matched_label = TryFindCells(label_dict, cellid, count)
                new_label = np.repeat(0, len(matched_label))
                for i, x in enumerate(np.unique(matched_label)):
                    new_label[matched_label == x] = i
                cell_type = [label_map[x] for x in np.unique(matched_label)]
                dataset = GeneExpressionDataset(
                    *GeneExpressionDataset.get_attributes_from_matrix(new_count, labels=new_label),
                    gene_names=geneid, cell_types=cell_type)
                print(dataset.X.shape, len(dataset.labels))
                if len(regev_data) > 0:
                    regev_data = GeneExpressionDataset.concat_datasets(regev_data, dataset)
                else:
                    regev_data = dataset

            dump_svmlight_file(regev_data.X, np.concatenate(regev_data.labels), 'regev_data.svmlight')
            np.save(self.save_path + 'regev_data.celltypes.npy', regev_data.cell_types)
            np.save(self.save_path + 'regev_data.gene_names.npy', regev_data.gene_names)
            count, labels = load_svmlight_file(self.save_path + 'regev_data.svmlight')
            cell_type = np.load(self.save_path + 'regev_data.celltypes.npy')
            gene_names = np.load(self.save_path + 'regev_data.gene_names.npy')
            return(count,labels,cell_type,gene_names)


