import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from config.definitions import ROOT_DIR
import os

# load the image

# print(np.asarray(img))

def main():
    img = Image.open(os.path.join(ROOT_DIR, 'supplimentary_materials', 'einstein.jpg'))

    # ar = np.asarray(img)
    # print

    def check_dim():
        print(f"The dimensions of the image is {np.asarray(img).shape}")
    check_dim()

    def check_is_integer():
        array_img = np.asarray(img)

        print(isinstance(array_img[0,0], int))

    check_is_integer()

    def min_max():
        array_img = np.asarray(img)

        print(f"Max value is :{np.max(array_img)}")
        print(f"Min value is :{np.min(array_img)}")

    min_max()

    def plot_img():
        plt.imshow(img)
        plt.show()

    # plot_img()

    def crop_img():
        array_img = np.asarray(img)

        new_img = array_img[100:-70,100:-70]

        plt.imshow(new_img)
        plt.show()

    # crop_img()

    def edit_img():
        array_img = np.asarray(img)
        blank_img = np.zeros_like((array_img))
        blank_img[100:-70,100:-70] = array_img[100:-70,100:-70]
        # print(array_img)
        # print(blank_img)

        plt.imshow(blank_img)
        plt.show()

    # edit_img()

    def flip_img():
        array_img = np.asarray(img)
        plt.imshow(np.flip(array_img))
        plt.show()

    # flip_img()

    def transpose_img():
        array_img = np.asarray(img).T
        plt.imshow(array_img)
        plt.show()

    # transpose_img()

    def h_flip():
        array_img = np.asarray(img)
        plt.imshow(np.fliplr(array_img))
        plt.show()

    # h_flip()

    def rot90():
        array_img = np.asarray(img)
        plt.imshow(np.rot90(array_img))
        plt.show()
    #
    # rot90()

    def inversion():
        array_img = np.asarray(img)
        plt.imshow(255 - array_img)
        plt.show()

    # inversion()

    def light_inversion():
        array_img = np.asarray(img)
        plt.imshow(array_img- 50)
        plt.show()

    # light_inversion()

    def warhol():
        array_img = np.asarray(img)

        # left right
        top_right = 50 - array_img

        # bottom left
        bottom_left = 100 - array_img

        # bottom right
        bottom_right = array_img - 50

        new_img_1 = np.concatenate((array_img,top_right),axis=1)
        new_img_2 = np.concatenate((bottom_left,bottom_right),axis=1)


        plt.imshow(np.concatenate((new_img_1,new_img_2),axis=0))
        plt.show()

    warhol()

    def naive_bw():
        array_img = np.asarray(img)

        new_img = np.un

if __name__ == "__main__":
    main()