from __future__ import division  # makes it so 4/5 != 0 but = .8 (thanks to Jeff Lund for the tip)
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def colorImgExp():
    """
    Color image example
    """
    # load an image
    img = pl.imread("mystery.png")
    pl.imshow(img)
    plt.show()

    # get one pixel value
    print ((img[0, 0, :]))

    # create some histograms
    plt.hist(img[:, 0], bins=256)
    plt.title("Red Histogram")
    plt.xlabel("Red intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    plt.hist(img[:, 1], bins=256)
    plt.title("Green Histogram")
    plt.xlabel("Green intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    plt.hist(img[:, 2], bins=256)
    plt.title("Blue Histogram")
    plt.xlabel("Blue intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    # do pixel-by-pixel negation
    negImg = np.zeros(img.shape)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            negImg[r, c] = 1.0 - img[r, c]

    # store it in a new image
    pl.imshow(negImg)
    plt.show()


def grayImgExp():
    """
    Gray image examples. Credit it to Tim Price for some of the ideas
    listed below.
    """
    # load color image
    img = pl.imread("mystery.png")

    # convert it to grayscale
    grayImg = (0.299 * img[:, :, 0]) + (0.587 * img[:, :, 1]) + (0.144 * img[:, :, 2])
    grayImg *= 255
    grayImg = np.around(grayImg).astype(int)

    # show the image
    pl.imshow(grayImg, cmap=pl.cm.Greys_r)
    plt.show()


def main():
    colorImgExp()
    grayImgExp()

# boiler-plate code
if __name__ == "__main__":
    main()