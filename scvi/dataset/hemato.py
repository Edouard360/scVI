import pandas as pd
import numpy as np
from zipfile import ZipFile
from .dataset import GeneExpressionDataset
from pathlib import Path


class HematoDataset(GeneExpressionDataset):
    r""" Loads hemato dataset.

    This dataset with continuous gene expression variations from hematopoeitic progenitor cells [31] contains
    4,016 cells and 7,397 genes. We removed the library basal-bm1 which was of poor quality based on authors
    recommendation. We use their population balance analysis result as a potential function for differentiation.

    Args:
        :save_path: Save path of raw data file. Default: ``'data/'``.

    Examples:
        >>> gene_dataset = HematoDataset()

    """
    def __init__(self, save_path='data/HEMATO/'):
        self.save_path = save_path

        # file and datazip file
        self.urls = ["https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2388072&format=file&"
                     "file=GSM2388072%5Fbasal%5Fbone%5Fmarrow%2Eraw%5Fumifm%5Fcounts%2Ecsv%2Egz",
                     "https://github.com/romain-lopez/scVI-reproducibility/raw/master/additional/data.zip"]

        # Originally: GSM2388072_basal_bone_marrow.raw_umifm_counts.csv.gz
        self.download_names = ["bBM.raw_umifm_counts.csv.gz", "data.zip"]
        self.gene_names_filename = "bBM.filtered_gene_list.paper.txt"
        self.spring_and_pba_filename = "bBM.spring_and_pba.csv"

        expression_data, gene_names, labels, self.y_spring, self.y_spring = self.download_and_preprocess()

        super(HematoDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                expression_data,
                labels=labels),
            gene_names=gene_names)

    def preprocess(self):
        print("Preprocessing Hemato data")

        with ZipFile(self.save_path + 'data.zip', 'r') as zip:
            zip.extractall(path=Path(self.save_path).parent)
        raw_counts = pd.read_csv(self.save_path + self.download_names[0], compression='gzip')

        # remove this library to avoid dealing with batch effects
        raw_counts.drop(raw_counts.index[raw_counts["library_id"] == "basal_bm1"], inplace=True)

        spring_and_pba = pd.read_csv(self.save_path + self.spring_and_pba_filename)
        # with open(self.save_path + self.gene_names_filename) as f:
        #     gene_filter_list = f.read()
        #
        # gene_names = gene_filter_list.splitlines()
        gene_names = np.loadtxt(self.save_path + self.gene_names_filename, dtype=np.str)

        data = raw_counts.merge(spring_and_pba, how="inner")
        expression_data = data[gene_names]
        x_spring = data["x_spring"].values
        y_spring = data["y_spring"].values

        meta = data[["Potential", "Pr_Er", "Pr_Gr", "Pr_Ly", "Pr_DC", "Pr_Mk", "Pr_Mo", "Pr_Ba"]]

        def logit(p):
            p = np.copy(p.values)
            p[p == 0] = np.min(p[p > 0])
            p[p == 1] = np.max(p[p < 1])
            return np.log(p / (1 - p))
        labels = logit(meta.iloc[:, 2]) - logit(meta.iloc[:, 1])
        expression_data = expression_data.values

        print("Finished preprocessing Hemato data")
        return expression_data, gene_names, labels, x_spring, y_spring

class MyeloidDataset(GeneExpressionDataset):

    def __init__(self,save_path = 'data/Myeloid'):
        self.save_path = save_path
        self.urls = ["ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72857/suppl/GSE72857_umitab.txt.gz",
                     "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72857/suppl/GSE72857_experimental_design.txt.gz"]
        self.download_names = ["GSE72857_umitab.txt.gz","GSE72857_experimental_design.txt.gz	"]


#
# import numpy as np
# gene_order = np.genfromtxt('Tusi/genes.csv',delimiter='')
