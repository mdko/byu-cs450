import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt


def colorImgExp():
    """
    Color image example
    """
    # load an image
    img = pil.open("mystery.png")
    img.show()

    # get one pixel value
    img.getpixel((3, 3))

    # store all pixels in 1D array len(imgData) = img.size[0] * img.size[1]
    # = width * height
    imgData = img.getdata()
    # throw the pixel data into a numpy array
    imgPix = np.array(imgData)

    # create some histograms
    plt.hist(imgPix[:, 0], bins=256)
    plt.title("Red Histogram")
    plt.xlabel("Red intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    plt.hist(imgPix[:, 1], bins=256)
    plt.title("Green Histogram")
    plt.xlabel("Green intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    plt.hist(imgPix[:, 2], bins=256)
    plt.title("Blue Histogram")
    plt.xlabel("Blue intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    # reshape the image pixels into 2D
    imgPix = np.resize(imgPix, (img.size[0], img.size[1], 3))

    # do pixel-by-pixel negation
    negImgPix = np.zeros((img.size[0], img.size[1], 3))
    for r in range(img.size[0]):
        for c in range(img.size[1]):
            negImgPix[r, c] = 255 - imgPix[r, c]

    # store it in a new image
    negImgPix = np.resize(negImgPix, (img.size[0] * img.size[1], 3))
    negImg = pil.new("RGB", img.size)

    # check out http://www.secnetix.de/olli/Python/list_comprehensions.hawk
    # for a better idea of what the following line is doing if it
    # seems a little strange
    negImg.putdata([tuple(color.astype(int)) for color in negImgPix])

    negImg.show()


def grayImgExp():
    """
    Gray image examples
    """
    # load color image
    img = pil.open("mystery.png")
    imgData = img.getdata()
    imgPix = np.array(imgData)

    # make it grayscale
    grayImgPix = np.around(np.dot(imgPix[..., :3],
        [0.299, 0.587, 0.144])).astype(int)
    plt.hist(grayImgPix, bins=256)
    plt.title("Gray Level Histogram")
    plt.xlabel("Gray intensities")
    plt.ylabel("Number of pixels")
    plt.show()

    # store it back in a new image
    grayImg = pil.new("L", img.size)
    grayImg.putdata(grayImgPix)
    grayImg.show()
    #grayImg.save("test.png")


def main():
    colorImgExp()
    grayImgExp()


# boiler-plate code
if __name__ == "__main__":
    main()