import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from bucky import *
import scipy.io
import os
from config.definitions import ROOT_DIR

def ex_1(theta_x, theta_y, theta_z):
    r_x = np.array([[1,0,0],
                    [0,np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    r_y = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)],
                    [0,1,0],
                    [np.sin(theta_y), 0, np.cos(theta_y)]])

    r_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0,1,0]])

    return r_x, r_y, r_z

def ex_3(theta_x, theta_y, theta_z):
    r_x, r_y, r_z = ex_1(theta_x, theta_y, theta_z)

    return r_x @ r_y @ r_z

def ex_5():
    # define the vertices and adjacency matrix
    vertices = np.array([[1,1,1],
                         [-1,1,1],
                         [1,-1,1],
                         [1,1,-1],
                         [-1,-1,1],
                         [-1,1,-1],
                         [1,-1,-1],
                         [-1,-1,-1]])

    edges = np.zeros((8,8))
    for i,j in zip([1,1,1,2,2,3,3,4,4,5,6,7],[2,3,4,5,6,5,7,6,7,8,8,8]):
        edges[i-1,j-1] = 1

    return vertices, edges

# function to plot the vertices and edges in 3d
def plot_vertices():
    vertices, edges = ex_5()

    # get the data
    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]

    # creating the figure
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection='3d')

    # create plot
    scat = ax.scatter3D(x,y,z,
                        alpha=0.7,
                        )

    for i in range(len(edges)):
        for j in range(len(edges)):
            if edges[i][j] == 1:
                source = vertices[i,:]
                target = vertices[j,:]
                x = [source[0], target[0]]
                y = [ source[1], target[1]]
                z = [ source[2], target[2]]

                ax.plot(x,y,z, color='black')

    ax.set_xlabel('x-axis', fontweight ='bold')
    ax.set_ylabel('y-axis', fontweight ='bold')
    ax.set_zlabel('z-axis', fontweight ='bold')
    plt.show()
def ex_6():
    theta_x, theta_y, theta_z = np.pi/3, np.pi/4, np.pi/6
    rotmat = ex_3(theta_x, theta_y, theta_z)
    return rotmat

def ex_7():
    vertices, edges = ex_5()
    rotmat = ex_6()

    return vertices @ rotmat

def ex_8():
    vertices, edges = ex_5()

    vert_rot = ex_7()
    print(vert_rot)

    plt.figure(figsize=(12,8))

    for i in range(8):
        for j in range(i, 8):
            if edges[i][j] == 1:
                source = vert_rot[i, :]
                target = vert_rot[j, :]
                x = [source[0], target[0]]
                y = [source[1], target[1]]

                plt.plot(x, y, color='black')

    plt.show()

def ex_9():
    # drop the second coord and the re-run the logic again
    vertices, edges = ex_5()

    vert_rot = ex_7()

    plt.figure(figsize=(12, 8))

    for i in range(8):
        for j in range(i, 8):
            if edges[i][j] == 1:
                source = vert_rot[i, :]
                target = vert_rot[j, :]
                x = [source[0], target[0]]
                y = [source[2], target[2]] # made changes here

                plt.plot(x, y, color='black')

    plt.show()

def ex_10():
# cannot get the data for the buckyball
    pass


# path to load the mat file
f_path = os.path.join(ROOT_DIR, "supplimentary_materials","f.mat")
v_path = os.path.join(ROOT_DIR, "supplimentary_materials","v.mat")

def ex_13():
    # load the vertices (v.mat)
    v = scipy.io.loadmat(v_path)['v']

    # load the triangular face of model (f.mat)
    f = scipy.io.loadmat(f_path)['f']

    return v, f-1

def ex_14():
    v, f = ex_13()
    shape = f.shape
    mFaces = shape[0]
    nFace = shape[1]
    return mFaces, nFace

def ex_15():
    v,f = ex_13()
    mFaces, nFace = ex_14()
    print(mFaces, nFace)

    # creating the figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')

    # plot the data
    for x,i in enumerate(range(mFaces)):
        # this ok
        ax.plot([v[f[i,0],0], v[f[i,1],0]],
                [v[f[i,0],1], v[f[i,1],1]],
                [v[f[i,0],2], v[f[i,1],2]],color='black')


        ax.plot([v[f[i, 0], 0], v[f[i, 2], 0]],
                [v[f[i, 0], 1], v[f[i, 2], 1]],
                [v[f[i, 0], 2], v[f[i, 2], 2]], color='black')

        ax.plot([v[f[i, 1], 0], v[f[i, 2], 0]],
                [v[f[i, 1], 1], v[f[i, 2], 1]],
                [v[f[i, 1], 2], v[f[i, 2], 2]], color='black')

    plt.show()

def ex_16():
    # compress the 3d image into 2d
    v, f = ex_13()

    mFaces, nFace = ex_14()

    rotation = ex_6()

    rotmat = v @ rotation
    rotmat = rotmat[:,:2]

    plt.figure(figsize=(12, 8))

    for i in range(mFaces):
        plt.plot([rotmat[f[i,0],0], rotmat[f[i,1],0]], [rotmat[f[i,0],1], rotmat[f[i,1],1]], color='black')
        plt.plot([rotmat[f[i, 0], 0], rotmat[f[i, 2], 0]], [rotmat[f[i, 0], 1], rotmat[f[i, 2], 1]], color='black')
        plt.plot([rotmat[f[i, 1], 0], rotmat[f[i, 2], 0]], [rotmat[f[i, 1], 1], rotmat[f[i, 2], 1]], color='black')

    plt.show()

def ex_17():
    # define the angles
    theta_x, theta_y, theta_z = -np.pi/3, 0, np.pi/4
    rotmat2 = ex_3(theta_x, theta_y, theta_z)

    v, f = ex_13()

    mFaces, nFace = ex_14()

    v_rot = v @ rotmat2
    v_proj = v_rot[:,:2]

    plt.figure(figsize=(12, 8))

    for i in range(mFaces):
        plt.plot([v_proj[f[i, 0], 0], v_proj[f[i, 1], 0]], [v_proj[f[i, 0], 1], v_proj[f[i, 1], 1]], color='black')
        plt.plot([v_proj[f[i, 0], 0], v_proj[f[i, 2], 0]], [v_proj[f[i, 0], 1], v_proj[f[i, 2], 1]], color='black')
        plt.plot([v_proj[f[i, 1], 0], v_proj[f[i, 2], 0]], [v_proj[f[i, 1], 1], v_proj[f[i, 2], 1]], color='black')

    plt.show()


if __name__ == "__main__":
    ex_17()