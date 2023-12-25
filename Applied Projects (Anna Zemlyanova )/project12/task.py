import numpy as np
from scipy.io import loadmat

# load the adj matrix
adj_mat = loadmat(r"/Applied Projects (Anna Zemlyanova )/supplimentary_materials/AdjMatrix.mat")["AdjMatrix"]

print(adj_mat)