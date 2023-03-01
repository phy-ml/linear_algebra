import numpy as np

def distance(a,b):
    '''calculates the straight line distance between two points a and b'''
    SS = sum(map(lambda x,y:(x-y)**2,a,b))
    return SS**0.5

def buckyball_vertices():
    data = np.genfromtxt("C60.csv", delimiter=',')
    data = data[1:, :]
    return data

def adj_mat():
    data = np.loadtxt("adj_mat.txt",dtype=int, delimiter=',')

    # create a 60 by 60 matrix
    adj = np.zeros((60,60))

    for i in data:
        x = i[0]-1
        y = i[1]-1
        adj[x][y] = 1

    return adj




vertices = buckyball_vertices()
adj_mat()