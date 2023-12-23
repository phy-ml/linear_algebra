import numpy as np
import matplotlib.pyplot as plt
import os


# read and store the images as matrix

def ex_1():
    db_path = r"E:\PycharmProjects\linear_algebra\supplimentary_materials\database"
    # fig, axs = plt.subplots(ncols=6, nrows=6)
    blank_lst = []

    # axs = axs.ravel()
    for num, i in enumerate(os.listdir(db_path)):
        if i.split("30altered")[0] == "person":
            continue
        else:
            img = plt.imread(db_path+"\\"+i)#.reshape(1,-1)
            blank_lst.append(img)

    img_mat = np.array(blank_lst).reshape(31,-1).T
    # print(img_mat.shape)

    # plt.imshow(img_mat[:,0].reshape(112,92))
    # plt.show()

    return img_mat

def ex_2():
    img_mat = ex_1()

    # compute the mean for each image row-wise
    mean_mat = img_mat.mean(axis=1)

    # plot the mean face
    # plt.imshow(mean_mat.reshape(112,92))
    # plt.show()

    return mean_mat.reshape(-1,1)

def ex_3():
    # substract the mean from all the face data
    face_mat = ex_1()
    mean_mat = ex_2()

    std_img = face_mat - mean_mat

    # plot the standardized images
    # plt.imshow(std_img[:,0].reshape(112,92))
    # plt.imshow(face_mat[:,0].reshape(112,92))
    # plt.show()
    # print(std_img.shape)

    return std_img

def ex_4():
    std_img = ex_3()

    #compute the co-variance matrix
    process_mat = std_img.T @ std_img

    # compute the eigen values and vectors
    eigen_val, eigen_vec = np.linalg.eig(process_mat)

    # print(eigen_val, eigen_vec.shape)

    return eigen_val, eigen_vec

def ex_5():
    # load the previous calculations
    std_img = ex_3()
    eigen_val, eigen_vec = ex_4()

    eigen_data = std_img @ eigen_vec

    # print(eigen_vec.shape)
    # print(eigen_val.shape)
    # print(std_img.shape)
    # print(eigen_data.shape)
    mean_mat = ex_2()
    # print(mean_mat.shape)
    m,n = 112, 92

    fig, axs = plt.subplots(ncols=5, nrows=6)
    axs = axs.ravel()

    for i in range(30):
        eigen_face = (eigen_data[:,i].reshape(-1,1) + mean_mat).reshape(m,n)

        axs[i].imshow(eigen_face)

    plt.show()


if __name__ == "__main__":
    ex_5()
