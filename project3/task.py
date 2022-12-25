import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from config.definitions import ROOT_DIR

img_path = os.path.join(ROOT_DIR, 'supplimentary_materials','daylilies.jpg')
img = Image.open(img_path)


def ex_1():
    plt.imshow(img)
    plt.show()

def ex_2(img):
    img = np.asarray(img)
    print(img.shape)

def ex_3(img):
    img = np.asarray(img)

    red_img = img[:,:,0]
    green_img = img[:, :, 1]
    blue_img = img[:, :, 2]

    fig, axs = plt.subplots(3)
    axs[0].imshow(red_img)
    axs[1].imshow(green_img)
    axs[2].imshow(blue_img)

    plt.show()

def ex_4(img):
    gray_matrix = np.array([[1/3, 1/3, 1/3],
                             [1/3, 1/3, 1/3],
                             [1/3, 1/3, 1/3]])

    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_gray = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel_color = img[i,j,:]
            img_gray[i,j,:] = (pixel_color@gray_matrix).astype(np.uint8)#.reshape(3,1)

    plt.imshow(img_gray)
    plt.show()

def ex_5(img):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])

    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_sepia = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel_color = img[i, j, :]
            img_sepia[i, j, :] = (pixel_color @ sepia_matrix).astype(np.uint8)  # .reshape(3,1)

    plt.imshow(img_sepia)
    plt.show()

def ex_6(img):
    red_matrix = np.array([[1,0,0],
                             [0,0,0],
                             [0,0,0]])

    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_red = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel_color = img[i, j, :]
            img_red[i, j, :] = (pixel_color @ red_matrix).astype(np.uint8)  # .reshape(3,1)

    plt.imshow(img_red)
    plt.show()

def ex_7(img,degree=60):
    permute_matrix = np.array([[0,0,1],
                             [0,1,0],
                             [1,0,0]])

    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_permute = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel_color = img[i, j, :]
            img_permute[i, j, :] = (pixel_color @ permute_matrix).astype(np.uint8)  # .reshape(3,1)

    plt.imshow(img_permute)
    plt.show()

    # hue rotate matrix
    mat_1 = np.array([[0.213, 0.715, 0.072],
                      [0.213, 0.715, 0.072],
                      [0.213, 0.715, 0.072]])

    mat_2 = np.cos(degree) * np.array([[0.787, -0.715, -0.072],
                                       [-0.213, 0.285, -0.072],
                                       [-0.213, -0.715, 0.928]])

    mat_3 = np.sin(degree) * np.array([[0.213, -0.715, 0.928],
                                       [0.143, 0.140, -0.283],
                                       [-0.787, 0.715, 0.072]])

    matrix = mat_1 + mat_2 + mat_3


    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_hue = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel_color = img[i, j, :]
            img_hue[i, j, :] = (pixel_color @ matrix).astype(np.uint8)  # .reshape(3,1)

    plt.imshow(img_hue)
    plt.show()

def ex_8(img):
    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_new = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel = img[i,j,:]
            # print(pixel)
            if pixel[-1] > 0:
                img_new[i,j,:] = [0,0,0]
            else:
                img_new[i,j,:] = pixel

    plt.imshow(img_new)
    plt.show()

def ex_9(img):
    img = np.asarray(img)

    # invert the image color
    invert_img = 255 - img
    print(invert_img)
    plt.imshow(invert_img)
    plt.show()

def ex_10(img):
    img = np.asarray(img)

    saturate = np.array([[1.2,0,0],
                         [0,0.75,0],
                         [0,0,2]])

    length, width, channel = img.shape

    # create blank image with zeros
    img_new = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel = img[i, j, :]
            img_new[i,j,:] = (pixel@saturate).astype(np.uint8)

    plt.imshow(img_new)
    plt.show()


def ex_11(img):
    img = np.asarray(img)

    user = np.array([[0.7, 0.15, 0.15],
                         [0.15, 0.7, 0.15],
                         [0.15, 0.15, 0.7]])

    length, width, channel = img.shape

    # create blank image with zeros
    img_new = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel = img[i, j, :]
            img_new[i, j, :] = (pixel @ user).astype(np.uint8)

    plt.imshow(img_new)
    plt.show()

def ex_12(img):
    img = np.asarray(img)
    user = np.array([[0.7, 0.15, 0.15],
                     [0.15, 0.7, 0.15],
                     [0.15, 0.15, 0.7]])

    length, width, channel = img.shape

    # create blank image with zeros
    img_new = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel = img[i, j, :]
            img_new[i, j, :] = (pixel @ user).astype(np.uint8)

    # get the inverse of the user matrix
    user_inv = np.linalg.inv(user)

    # blank image
    inv_img = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel = img_new[i, j, :]
            inv_img[i, j, :] = (pixel @ user_inv).astype(np.uint8)

    fig, axs = plt.subplots(3)
    axs[0].imshow(img)
    axs[1].imshow(img_new)
    axs[2].imshow(inv_img)

    plt.show()

def ex_13(img):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    img = np.asarray(img)
    length, width, channel = img.shape

    # create blank image with zeros
    img_sepia = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel_color = img[i, j, :]
            img_sepia[i, j, :] = (pixel_color @ sepia_matrix).astype(np.uint8)

    # inverse image
    user_inv = np.linalg.inv(sepia_matrix)

    # blank image
    inv_img = np.zeros_like(img)

    for i in range(length):
        for j in range(width):
            pixel = img_sepia[i, j, :]
            inv_img[i, j, :] = (pixel @ user_inv).astype(np.uint8)

    fig, axs = plt.subplots(3)
    axs[0].imshow(img)
    axs[1].imshow(img_sepia)
    axs[2].imshow(inv_img)

    plt.show()

def ex_14(img):
    img = np.asarray(img)

    # gamma 1
    gamma_1 = (img**(0.9) + 30).astype(np.uint8)

    # gamma 2
    gamma_2 = (img**(1.1) - 50).astype(np.uint8)

    fig, axs = plt.subplots(3)
    axs[0].imshow(img)
    axs[1].imshow(gamma_1)
    axs[2].imshow(gamma_2)

    plt.show()


if __name__ == "__main__":
    ex_14(img)
