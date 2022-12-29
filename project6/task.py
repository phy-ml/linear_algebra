import numpy as np
from PIL import Image
import os
from config.definitions import ROOT_DIR
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

img = Image.open(os.path.join(ROOT_DIR,'supplimentary_materials','einstein.jpg'))

def ex_1(img):
    img = np.asarray(img)
    m,n = img.shape
    return m,n

def ex_2(img):
    array = np.asarray(img)
    m,n = array.shape

    # add noise to the image
    noise = 50*np.random.random(size=(m,n)) - 0.5

    noise_img = (array + noise).astype(np.uint8)

    # plot the original and image with nosie
    # fig, axs = plt.subplots(2)
    # axs[0].imshow(img)
    # axs[1].imshow(noise_img)
    # plt.show()

    return noise_img

def ex_4():
    kavg_1 = (1/5) * np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]])

    kavg_2 = (1/9) * np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])

    return kavg_1, kavg_2

def ex_5(img):
    # array = np.asarray(img)

    noise_img = ex_2(img)

    k1, k2 = ex_4()

    # convolve the noise image with both the kernels
    img1 = convolve2d(noise_img,k1)
    img2 = convolve2d(noise_img,k2)

    # plot the original and image with nosie
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(img)
    axs[0,0].set_title('Original Image')

    axs[0,1].imshow(noise_img)
    axs[0,1].set_title('Noisy Image')

    axs[1,0].imshow(img1)
    axs[1, 0].set_title('Average Kernel 1')

    axs[1,1].imshow(img2)
    axs[1, 1].set_title('Average Kernel 2')

    plt.show()

def ex_6(img):
    return 1/5

if __name__ == "__main__":
    print(ex_5(img))