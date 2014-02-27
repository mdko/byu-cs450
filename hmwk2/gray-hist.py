import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt


def histogramEqualization():

    # load color image
    img = pil.open("mystery.png")
    imgData = img.getdata()
    imgPix = np.array(imgData)

    grayImgPix = np.around(np.dot(imgPix[..., :3],
        [0.299, 0.587, 0.144])).astype(int)
    # Store the frequencies for each intensity, 0 - 255
    grays = np.zeros(256, int)
    for r in range(grayImgPix.size):        # for each pixel in 1-d long array
            grayVal = grayImgPix[r]
            if (grayVal > 255):
                grayVal = 255
            grays[grayVal] = grays[grayVal] + 1
    # Store each intensity's probability density
    pdf = np.zeros(256, np.float64)
    numPixs = img.size[0] * img.size[1]
    for r in range(pdf.size):
        pdf[r] = np.float64(grays[r] / np.float64(numPixs))
    # assert summation pdf from 0 to r = 1.0--verified it does

    # Store the transformation function values, equalization histogram lookup table
    transVals = np.zeros(256, np.float64)
    histSum = np.zeros(256)
    for r in range(pdf.size):
        prev = None
        if (r == 0.0):
            prev = 0.0
        else:
            prev = transVals[r - 1]
        transVals[r] = (255.0 * pdf[r]) + prev

        # find the rounded value of the current value we just computed
        rounded = int(round(transVals[r]))
        # be remembering what they map to to store in new histogram
        histSum[rounded] = histSum[rounded] + grays[r]

        #print (transVals[r])
        #print (rounded)
        #print (histSum[rounded])

    # Store the equalized pdf
    equalPdf = np.zeros(256, np.float64)
    a = 0.0
    for r in range(equalPdf.size):
        equalPdf[r] = np.float64(histSum[r] / np.float64(numPixs))
        print(equalPdf[r])
        a += equalPdf[r]
    # check to make sure = 1.0
    #print (a)

    plt.bar(list(range(transVals.size)), transVals)
    plt.title("Transformation Function")
    plt.xlabel("rk")
    plt.ylabel("sk")
    plt.show()

    plt.bar(list(range(equalPdf.size)), equalPdf)
    plt.title("Gray Level Histogram")
    plt.xlabel("Gray intensities")
    plt.ylabel("Proportion of pixels")
    plt.show()

    # store it back in a new image
    for r in range(grayImgPix.size):
        # print (transVals[grayImgPix[r]])
        grayVal = grayImgPix[r]
        if (grayVal > 255):  # sometimes 263 pops up...
            grayVal = 255
        grayImgPix[r] = transVals[grayVal]
    newGrayImg = pil.new("L", img.size)
    newGrayImg.putdata(grayImgPix)
    newGrayImg.show()
    newGrayImg.save("equalized-mystery.png")

    # print out transformation lookup table
    for c in range(transVals.size):
        print ("%i | %f" % (c, int(round(transVals[c]))))


def grayImgExp():
    # do stuff
    # load color image
    img = pil.open("mystery.png")
    imgData = img.getdata()
    imgPix = np.array(imgData)
    numPixs = img.size[0] * img.size[1]

    # make it grayscale, print out histogram manually
    grayImgPix = np.around(np.dot(imgPix[..., :3],
        [0.299, 0.587, 0.144])).astype(int)
    grays = np.zeros(256, int)
    for r in range(grayImgPix.size):        # for each pixel in 1-d long array
            grayVal = grayImgPix[r]
            if (grayVal > 255):
                grayVal = 255
            grays[grayVal] = grays[grayVal] + 1
    # probability distribution
    pdf = np.zeros(256, np.float64)
    for r in range(pdf.size):
        pdf[r] = np.float64(grays[r] / np.float64(numPixs))

    plt.bar(list(range(pdf.size)), pdf)
    plt.title("Gray Level Histogram")
    plt.xlabel("Gray intensities")
    plt.ylabel("Proportion of pixels")
    plt.show()


def main():
    grayImgExp()
    histogramEqualization()


# boiler-plate code
if __name__ == "__main__":
    main()