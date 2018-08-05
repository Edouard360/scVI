import numpy as np

cellTypeProb = np.genfromtxt('bBM/text_file_output/B.csv', delimiter=',')
# cellType = np.genfromtxt('bBM/fate_labels.csv', delimiter=',', dtype='str')
cellType = ['Erythroid', 'Granulocytic', 'Lymphocytic', 'Dendritic', 'Megakaryocytic','Monocytic','Basophilic']
pseudoTime = np.genfromtxt('bBM/text_file_output/V.txt')
X = np.genfromtxt('raw.umi.csv', delimiter=',')
meta = np.genfromtxt('raw.meta.txt', delimiter=',',dtype='str')
meta = meta[1:, ]
X = X[meta[:, 4] == '1', :]
meta = meta[meta[:, 4] == '1', :]
pseudoTime[cellTypeProb[:,0]>0.1]
