from scvi.harmonization.utils_chenling import get_matrix_from_h5, TryFindCells
import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import os
from scipy.sparse import csr_matrix

def combine_MacoskoRegev(ngenes=5000):
    dataset1 = MacoskoDataset()
    dataset2 = RegevDataset()
    gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
    gene_dataset.subsample_genes(ngenes)
    keys = gene_dataset.cell_types
    labels = np.concatenate(gene_dataset.labels)
    key_color_order = ['Pvalb low', 'Pvalb', 'Pvalb 1', 'Pvalb 2',
                       'Pvalb Ex_1', 'Pvalb Ex_2', 'Pvalb Ex',
                       'Pvalb Astro_1', 'Pvalb Astro_2',
                       'L2/3 IT Astro', 'L2/3 IT Macc1', 'L2/3 IT Sla_Astro', 'L2/3 IT', 'L2/3 IT Sla',
                       'L2/3 IT Sla_Inh',
                       'Sst Tac2', 'Sst Myh8', 'Sst Etv1', 'Sst Chodl', 'Sst',
                       'L5 PT_2', 'L5 PT IT', 'L5 PT_1',
                       'L5 IT Tcap_1_3', 'L5 IT Tcap_2', 'L5 IT Tcap_Astro', 'L5 IT Tcap_1', 'L5 IT Tcap_L2/3',
                       'L5 IT Tcap_Foxp2', 'L5 IT Tcap_3',
                       'L5 IT Aldh1a7_2', 'L5 IT Aldh1a7', 'L5 IT Aldh1a7_1',
                       'L5 NP', 'L5 NP Slc17a8',
                       'L6 IT Car3', 'L6 CT Olig', 'L6 IT Maf', 'L6 IT Ntn5 Mgp', 'L6 IT Ntn5 Inpp4b',
                       'L6 CT Nxph2', 'L6 CT Astro', 'L6 CT', 'L6 CT Grp',
                       'L6b', 'L6b F2r',
                       'Lamp5 Sncg', 'Lamp5 Egln3', 'Lamp5 Slc35d3',
                       'Vip Rspo4', 'Vip Serpinf1', 'Vip',
                       'Astro Ex', 'Astro Aqp4',
                       'OPC Pdgfra',
                       'VLMC Osr1',
                       'Oligo Enpp6_1', 'Oligo Enpp6_2', 'Oligo Opalin',
                       'Sncg Ptprk',
                       'Endo Slc38a5', 'Endo Slc38a5_Peri_2', 'Endo Slc38a5_Peri_1']
    key_color_order = [x.upper() for x in key_color_order]
    clust_large = np.concatenate([np.repeat(0, 4),  # Pvalb
                                  np.repeat(1, 3),  # Pvalb Ex
                                  np.repeat(2, 2),  # Pvalb Astrol
                                  np.repeat(3, 6),  # L2/3
                                  np.repeat(4, 5),  # Sst
                                  np.repeat(5, 3),  # L5 PT
                                  np.repeat(6, 7),  # L5 IT Tcap_1_3
                                  np.repeat(7, 3),  # L5 IT Aldh1a7_2
                                  np.repeat(8, 2),  # L5 NP
                                  np.repeat(9, 5),  # L6 IT
                                  np.repeat(10, 4),  # L6 CT
                                  np.repeat(11, 2),  # L6b
                                  np.repeat(12, 3),  # Lamp5
                                  np.repeat(13, 3),  # VIP
                                  np.repeat(14, 2),  # Astro
                                  np.repeat(15, 1),  # OPC
                                  np.repeat(16, 1),  # VLMC
                                  np.repeat(17, 3),  # oligo
                                  np.repeat(18, 1),  # sncg
                                  np.repeat(19, 3)  # Endo
                                  ])
    label_dict = dict(zip(keys, np.arange(len(keys))))
    ordered_label = [label_dict[x] for x in key_color_order]
    label_dict = dict(zip(ordered_label, np.arange(len(ordered_label))))
    new_labels = np.asarray([label_dict[x] for x in labels])
    new_cell_types = key_color_order
    gene_dataset.labels = new_labels
    gene_dataset.cell_types = new_cell_types
    return (gene_dataset, clust_large)


class MacoskoDataset(GeneExpressionDataset):
    def __init__(self, save_path='../AIBS/'):
        self.save_path = save_path
        count, labels, cell_type, gene_names,labels_groups = self.preprocess()
        super(MacoskoDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)
        self.labels_groups = labels_groups
    def preprocess(self):
        if os.path.isfile(self.save_path + 'macosko_data.svmlight'):
            count, labels = load_svmlight_file(self.save_path + 'macosko_data.svmlight')
            cell_type = np.load(self.save_path + 'macosko_data.celltypes.npy')
            labels_groups = np.load(self.save_path + 'macosko_data.labels_groups.npy')
            gene_names = np.load(self.save_path + 'macosko_data.gene_names.npy')
            return(count, labels, cell_type, gene_names, labels_groups)
        else:
            macosko_batches = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            label = np.genfromtxt(self.save_path + '10X_nuclei_Macosko/cluster.membership.csv', dtype='str',
                                  delimiter=',')
            label_batch = np.asarray([str(int(int(x.split('-')[1].split('"')[0]) / 11)) for x in label[1:, 0]])
            label_barcode = np.asarray([x.split('-')[0].split('"')[1] for x in label[1:, 0]])
            label_cluster = np.asarray([x.split('"')[1] for x in label[1:, 1]])
            label_map = np.genfromtxt(self.save_path + '10X_nuclei_Macosko/cluster.annotation.csv', dtype='str',
                                      delimiter=',')
            label_map = dict(
                zip([x.split('"')[1] for x in label_map[:, 0]], [x.split('"')[1] for x in label_map[:, 1]]))
            macosko_data = []
            for batch_i, batch in enumerate(macosko_batches):
                geneid, cellid, count = get_matrix_from_h5(
                    self.save_path + '10X_nuclei_Macosko/' + '171218_p56m1' + batch + '/outs/filtered_gene_bc_matrices_h5.h5',
                    'mm10_premrna')
                count = count.T.tocsr()
                print(count.shape, len(geneid), len(cellid))
                cellid = [id.split('-')[0] for id in cellid]
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
                if len(macosko_data) > 0:
                    macosko_data = GeneExpressionDataset.concat_datasets(macosko_data, dataset)
                else:
                    macosko_data = dataset
            dataset = macosko_data
            cell_type = dataset.cell_types
            groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
                      'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
                      'SMC', 'MICRO']
            cell_type = [x.upper() for x in cell_type]
            groups = [x.upper() for x in groups]
            labels = np.asarray([cell_type[x] for x in np.concatenate(dataset.labels)])
            cell_type_bygroup = np.concatenate([[x for x in cell_type if x.startswith(y)] for y in groups])
            new_labels_dict = dict(zip(cell_type_bygroup, np.arange(len(cell_type_bygroup))))
            new_labels = np.asarray([new_labels_dict[x] for x in labels])
            labels_groups = [[i for i, x in enumerate(groups) if y.startswith(x)][0] for y in cell_type_bygroup]
            dump_svmlight_file(dataset.X, new_labels, self.save_path + 'macosko_data.svmlight')
            np.save(self.save_path + 'macosko_data.celltypes.npy', cell_type_bygroup)
            np.save(self.save_path + 'macosko_data.gene_names.npy', dataset.gene_names)
            np.save(self.save_path + 'macosko_data.labels_groups.npy', labels_groups)
            return (dataset.X, new_labels, cell_type_bygroup, dataset.gene_names, labels_groups)


class RegevDataset(GeneExpressionDataset):
    def __init__(self, save_path='../AIBS/'):
        self.save_path = save_path
        count, labels, cell_type, gene_names, labels_groups = self.preprocess()
        super(RegevDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)
        self.labels_groups = labels_groups

    def preprocess(self):
        if os.path.isfile(self.save_path + 'regev_data.svmlight'):
            count, labels = load_svmlight_file(self.save_path + 'regev_data.svmlight')
            cell_type = np.load(self.save_path + 'regev_data.celltypes.npy')
            gene_names = np.load(self.save_path + 'regev_data.gene_names.npy')
            labels_groups = np.load(self.save_path + 'regev_data.labels_groups.npy')
            return(count, labels, cell_type, gene_names, labels_groups)
        else:
            regev_batches = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            label = np.genfromtxt(self.save_path + '10X_nuclei_Regev/cluster.membership.csv', dtype='str',
                                  delimiter=',')
            label_batch = np.asarray([str(int(int(x.split('-')[1].split('"')[0]))) for x in label[1:, 0]])
            label_barcode = np.asarray([x.split('-')[0].split('"')[1] for x in label[1:, 0]])
            label_cluster = np.asarray([x.split('"')[1] for x in label[1:, 1]])
            label_map = np.genfromtxt(self.save_path + '10X_nuclei_Regev/cluster.annotation.csv', dtype='str',
                                      delimiter=',')
            label_map = dict(
                zip([x.split('"')[1] for x in label_map[:, 0]], [x.split('"')[1] for x in label_map[:, 1]]))
            regev_data = []
            for batch_i, batch in enumerate(regev_batches):
                geneid, cellid, count = get_matrix_from_h5(
                    self.save_path + '10X_nuclei_Regev/' + batch + '1/filtered_gene_bc_matrices_h5.h5',
                    'mm10-1.2.0_premrna')
                count = count.T.tocsr()
                cellid = [id.split('-')[0] for id in cellid]
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
            dataset = regev_data
            cell_type = dataset.cell_types
            groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
                      'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
                      'SMC', 'MICRO']
            cell_type = [x.upper() for x in cell_type]
            groups = [x.upper() for x in groups]
            labels = np.asarray([cell_type[x] for x in np.concatenate(dataset.labels)])
            cell_type_bygroup = np.concatenate([[x for x in cell_type if x.startswith(y)] for y in groups])
            new_labels_dict = dict(zip(cell_type_bygroup, np.arange(len(cell_type_bygroup))))
            new_labels = np.asarray([new_labels_dict[x] for x in labels])
            labels_groups = [[i for i, x in enumerate(groups) if y.startswith(x)][0] for y in cell_type_bygroup]
            dump_svmlight_file(dataset.X, new_labels, self.save_path+'regev_data.svmlight')
            np.save(self.save_path + 'regev_data.celltypes.npy', cell_type_bygroup)
            np.save(self.save_path + 'regev_data.gene_names.npy', dataset.gene_names)
            np.save(self.save_path + 'regev_data.labels_groups.npy', labels_groups)
            return (dataset.X, new_labels, cell_type_bygroup, dataset.gene_names, labels_groups)


class Zeng10X(GeneExpressionDataset):
    def __init__(self, save_path='../AIBS/',cell_compartment='nuclei'):
        self.save_path = save_path
        self.cell_compartment = cell_compartment
        count, labels, cell_type, gene_names, labels_groups = self.preprocess()
        super(Zeng10X, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)
        self.labels_groups = labels_groups

    def preprocess(self):
        if self.cell_compartment is 'cell':
            geneid, cellid, count = get_matrix_from_h5(self.save_path + '10X_cells_AIBS/umi_counts.h5',
                                                       'mm10-1.2.0_premrna')
            label = np.genfromtxt(self.save_path + '10X_cells_AIBS/cluster.membership.csv', dtype='str', delimiter=',')
            label_map = np.genfromtxt(self.save_path + '10X_cells_AIBS/cluster.annotation.csv', dtype='str',
                                      delimiter=',')
        elif self.cell_compartment is 'nuclei':
            geneid, cellid, count = get_matrix_from_h5(self.save_path + '10X_nuclei_AIBS/umi_counts.h5',
                                                       'mm10-1.2.0_premrna')
            label = np.genfromtxt(self.save_path + '10X_nuclei_AIBS/cluster.membership.csv', dtype='str', delimiter=',')
            label_map = np.genfromtxt(self.save_path + '10X_nuclei_AIBS/cluster.annotation.csv', dtype='str',
                                      delimiter=',')
        count = count.T.tocsr()
        label_cluster = np.asarray([x.split('"')[1] for x in label[1:, 1]])
        label_barcode = np.asarray([x.split('"')[1] for x in label[1:, 0]])
        label_barcode = np.asarray([x.split('L8TX')[0] for x in label_barcode])
        label_dict = dict(zip(label_barcode, label_cluster))
        new_count, matched_label = TryFindCells(label_dict, cellid, count)
        map_clust = np.asarray([x.split('"')[1] for x in label_map[1:, 0]])
        cell_type = np.asarray([x.split('"')[1] for x in label_map[1:, 1]])
        cell_type[cell_type == 'IT L6b'] = 'L6B IT'
        unique_index = np.unique(cell_type, return_index=True)[1]
        not_unique = [x for x in np.arange(len(cell_type)) if x not in set(unique_index)]
        for k, i in enumerate(not_unique):
            cell_type[i] = cell_type[i] + '_' + str(k)
        groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
        'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
        'SMC', 'MICRO']
        cell_type = [x.upper() for x in cell_type]
        groups = [x.upper() for x in groups]
        temp = dict(zip(map_clust, cell_type))
        labels = np.asarray([temp[x] for x in matched_label])
        cell_type_bygroup = np.concatenate([[x for x in cell_type if x.startswith(y)] for y in groups])
        new_labels_dict = dict(zip(cell_type_bygroup, np.arange(len(cell_type_bygroup))))
        new_labels = np.asarray([new_labels_dict[x] for x in labels])
        labels_groups = [[i for i, x in enumerate(groups) if y.startswith(x)][0] for y in cell_type_bygroup]
        return (new_count, new_labels, cell_type_bygroup, geneid, labels_groups)

class ZengSS2(GeneExpressionDataset):
    def __init__(self, save_path='../AIBS/',cell_compartment='nuclei'):
        self.save_path = save_path
        self.cell_compartment = cell_compartment
        count, labels, cell_type, gene_names, labels_groups = self.preprocess()
        super(ZengSS2, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)
        self.labels_groups = labels_groups

    def preprocess(self):
        if self.cell_compartment is 'cell':
            mat = np.genfromtxt(self.save_path + 'SmartSeq_cells_AIBS/exon.counts.csv',dtype='str',delimiter=',')
            label = np.genfromtxt(self.save_path + 'SmartSeq_cells_AIBS/cluster.membership.csv', dtype='str', delimiter=',')
            label_map = np.genfromtxt(self.save_path + 'SmartSeq_cells_AIBS/cluster.annotation.csv', dtype='str',
                                      delimiter=',')
            cellid = np.asarray([x.split('"')[1] for x in mat[0, 1:]])
            geneid = np.asarray([x.split('"')[1] for x in mat[1:, 0]])
            count = mat[1:,1:].astype('int')
            count = csr_matrix(count.T)
        elif self.cell_compartment is 'nuclei':
            mat = np.genfromtxt(self.save_path + 'SmartSeq_nuclei_AIBS/exon.counts.csv',dtype='str',delimiter=',')
            label = np.genfromtxt(self.save_path + 'SmartSeq_nuclei_AIBS/cluster.membership.csv', dtype='str', delimiter=',')
            label_map = np.genfromtxt(self.save_path + 'SmartSeq_nuclei_AIBS/cluster.annotation.csv', dtype='str',
                                      delimiter=',')
            cellid = np.asarray([x.split('"')[1] for x in mat[0, 1:]])
            geneid = np.asarray([x.split('"')[1] for x in mat[1:, 0]])
            count = mat[1:,1:].astype('int')
            count = csr_matrix(count.T)
        label_cluster = np.asarray([x.split('"')[1] for x in label[1:, 1]])
        label_barcode = np.asarray([x.split('"')[1] for x in label[1:, 0]])
        label_barcode = np.asarray([x.split('L8TX')[0] for x in label_barcode])
        label_dict = dict(zip(label_barcode, label_cluster))
        new_count, matched_label = TryFindCells(label_dict, cellid, count)
        map_clust = np.asarray([x.split('"')[1] for x in label_map[1:, 0]])
        cell_type = np.asarray([x.split('"')[1] for x in label_map[1:, 1]])
        groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
                  'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
                  'SMC','MICRO']
        cell_type = [x.upper() for x in cell_type]
        groups = [x.upper() for x in groups]
        temp = dict(zip(map_clust, cell_type))
        labels = np.asarray([temp[x] for x in matched_label])
        cell_type_bygroup = np.concatenate([[x for x in cell_type if x.startswith(y)] for y in groups])
        new_labels_dict = dict(zip(cell_type_bygroup, np.arange(len(cell_type_bygroup))))
        new_labels = np.asarray([new_labels_dict[x] for x in labels])
        labels_groups = [[i for i, x in enumerate(groups) if y.startswith(x)][0] for y in cell_type_bygroup]
        return (new_count, new_labels, cell_type_bygroup, geneid, labels_groups)

