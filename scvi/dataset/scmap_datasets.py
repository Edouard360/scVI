from scvi.dataset import LoomDataset


class MuraroDataset(LoomDataset):
    def __init__(self, save_path='data/'):
        super(MuraroDataset, self).__init__(filename='muraro.loom',
                                            save_path=save_path,
                                            url='https://github.com/YosefLab/scVI-data/raw/master/muraro.loom')


class BaronDataset(LoomDataset):
    def __init__(self, save_path='data/'):
        super(BaronDataset, self).__init__(filename='baron.loom',
                                           save_path=save_path,
                                           url='https://github.com/YosefLab/scVI-data/raw/master/baron.loom')


class SegerstolpeDataset(LoomDataset):
    def __init__(self, save_path='data/'):
        super(SegerstolpeDataset, self).__init__(filename='segerstolpe.loom',
                                                 save_path=save_path,
                                                 url='https://github.com/YosefLab/scVI-data/raw/master/segerstolpe.loom')


class XinDataset(LoomDataset):
    def __init__(self, save_path='data/'):
        super(XinDataset, self).__init__(filename='xin.loom',
                                         save_path=save_path,
                                         url='https://github.com/YosefLab/scVI-data/raw/master/xin.loom')


''' The script to run to generate the loom files.'''

# from scvi.harmonization.classification.scmap import SCMAP
# scmap = SCMAP()
# dataset_xin = scmap.create_dataset("../scmap/xin/xin.rds")
# cell_types_xin = list(filter(lambda p: ".contaminated" not in p, dataset_xin.cell_types))
# dataset_xin.filter_cell_types(cell_types_xin)
# dataset_xin.export_loom('xin.loom')
#
# dataset_segerstolpe = scmap.create_dataset("../scmap/segerstolpe/segerstolpe.rds")
# cell_types_segerstolpe = list(filter(lambda p: p != "not applicable", dataset_segerstolpe.cell_types))
# dataset_segerstolpe.filter_cell_types(cell_types_segerstolpe)
# all_cell_types = list(
#     filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"],
#            dataset_segerstolpe.cell_types))
# dataset_segerstolpe.filter_cell_types(all_cell_types)
# dataset_segerstolpe.export_loom('segerstolpe.loom')
#
# dataset_muraro = scmap.create_dataset("../scmap/muraro/muraro.rds")
# all_cell_types = list(
#     filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"],
#            dataset_muraro.cell_types))
# dataset_muraro.filter_cell_types(all_cell_types)
# dataset_muraro.export_loom('muraro.loom')
#
# dataset_baron = scmap.create_dataset("../scmap/baron-human/baron-human.rds")
# all_cell_types = list(
#     filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"],
#            dataset_baron.cell_types))
# dataset_baron.filter_cell_types(all_cell_types)
# dataset_baron.export_loom('baron.loom')
#
# dataset_shekhar = scmap.create_dataset("../scmap/shekhar/shekhar.rds")
# all_cell_types = list(
#     filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"],
#            dataset_shekhar.cell_types))
# dataset_shekhar.filter_cell_types(all_cell_types)
# dataset_shekhar.export_loom('shekhar.loom')
#
# dataset_macosko = scmap.create_dataset("../scmap/macosko/macosko.rds")
# all_cell_types = list(
#     filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"],
#            dataset_macosko.cell_types))
# dataset_macosko.filter_cell_types(all_cell_types)
# dataset_macosko.export_loom('macosko.loom')
'''

dataset = GeneExpressionDataset.concat_datasets(dataset_xin, dataset_segerstolpe, dataset_muraro, dataset_baron,
                                                on='gene_symbols')
dataset.export_pickle("../scmap/d4-all.pickle") # This is for experiments with the 4 datasets
dataset.subsample_genes(subset_genes=(dataset.X.max(axis=0) <= 2500).ravel())
dataset.subsample_genes(1500)
'''
