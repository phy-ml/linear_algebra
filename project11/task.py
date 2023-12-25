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
    # m,n = 112, 92
    #
    # fig, axs = plt.subplots(ncols=5, nrows=6)
    # axs = axs.ravel()
    #
    # for i in range(30):
    #     eigen_face = (eigen_data[:,i].reshape(-1,1) + mean_mat).reshape(m,n)
    #
    #     axs[i].imshow(eigen_face)
    #
    # plt.show

    return eigen_data


def ex_6():
    eigen_val, eigen_vec = ex_4()
    img_mat = ex_1()
    std_img = ex_3()

    # p_mat = std_img.T @ std_img

    # check if the p_mat is symmetric
    # print(np.allclose(p_mat, p_mat.T, rtol=1e-05, atol=1e-08))

    # check if the eigen vectors are orthogonal
    # orth_eign_vec = eigen_vec.T @ eigen_vec
    # # create an identify matrix
    # id_matrix = np.identity(len(orth_eign_vec))
    # print(np.allclose(orth_eign_vec, id_matrix, rtol=1e-05, atol=1e-08))

    # check if the product of eigen vector and std_matrix is diagonal matrix
    eigen_data = std_img @ eigen_vec
    product = eigen_data.T @ eigen_data

    # check if the product is a diagonal matrix or not
    # print(product.diagonal())
    check_mat = np.diag(product.diagonal())
    print(np.allclose(product, check_mat, rtol=1e-05, atol=1e-05))


def ex_7():
    # recognition of an altered image (in sunglasses)

    # read the altered-image
    alt_img = plt.imread("E:\PycharmProjects\linear_algebra\supplimentary_materials\database\person30altered1.pgm").reshape(-1,1)

    # plt.imshow(alt_img)
    # plt.show()

    # load computations from pervious steps
    std_img = ex_3()
    eigen_val, eigen_vec = ex_4()

    eigen_data = std_img @ eigen_vec
    mean_mat = ex_2()

    product = eigen_data.T @ eigen_data

    diff = mean_mat - alt_img
    w = eigen_data.T @ diff
    w = (w.ravel()/np.diag(product)).reshape(-1,1)

    u_approx = (eigen_data@w + mean_mat).reshape(112,92)
    u_original = alt_img.reshape(112,92)

    fig, axs = plt.subplots(2,1)
    axs = axs.ravel()

    axs[0].imshow(u_approx)
    axs[1].imshow(u_original)

    plt.show()

def ex_8():
    # read the altered-image
    alt_img = plt.imread(
        "E:\PycharmProjects\linear_algebra\supplimentary_materials\database\person30altered2.pgm").reshape(-1, 1)

    # plt.imshow(alt_img)
    # plt.show()

    # load computations from pervious steps
    std_img = ex_3()
    eigen_val, eigen_vec = ex_4()

    eigen_data = std_img @ eigen_vec
    mean_mat = ex_2()

    product = eigen_data.T @ eigen_data

    diff = mean_mat - alt_img
    w = eigen_data.T @ diff
    w = (w.ravel() / np.diag(product)).reshape(-1, 1)

    u_approx = (eigen_data @ w + mean_mat).reshape(112, 92)
    u_original = alt_img.reshape(112, 92)

    fig, axs = plt.subplots(2, 1)
    axs = axs.ravel()

    axs[0].imshow(u_approx)
    axs[1].imshow(u_original)

    plt.show()




if __name__ == "__main__":
    ex_8()
