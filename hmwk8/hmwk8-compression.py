import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import scipy.misc as sp
import math

def written():
    #Entropy for relative frequencies
    relative_frequencies = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702,
                            0.02228, 0.02015, 0.06094, 0.06966, 0.00153,
                            0.00772, 0.04025, 0.02406, 0.06749, 0.07507,
                            0.01929, 0.00095, 0.05987, 0.06327, 0.09056,
                            0.02758, 0.00978, 0.02360, 0.00150, 0.01974,
                            0.00074]

    #For debugging (should equal 1.00 or 0.99)
    #
    # total = 0.0
    # for freq in relative_frequencies:
    #     total += freq
    # print total

    sigma = 0.00
    for freq in relative_frequencies:
        sigma += freq * np.log2(freq)
    sigma = -sigma
    print "Entropy for English language with relative frequencies: %f" % sigma

    #Entropy if each letter occurred with same frequency
    sigma = 0.00
    freq = 1.0/26.0
    for a in range(26):
        sigma += freq * np.log2(freq)
    sigma = -sigma
    print "Entropy for English language with equal frequencies: %f" % sigma


def programming():
    files = [('whitebox-original.tif', 'whitebox-residual.tif')
             ,('cobra-original.tif', 'cobra-residual.tif')
             ,('parrots-original.tif', 'parrots-residual.tif')
             ,('ball-original.tif', 'ball-residual.tif')
             ,('blocks-original.tif', 'blocks-residual.tif')
             ,('gull-original.tif', 'gull-residual.tif')
             ]
    for f in files:
        displayHist(f[0])
        encode(f[0])
        displayHist(f[1])
        decode(f[1])    # to test my decoder works


def encode(filename):
    img = sp.imread(filename)
    img = img[:,:,]
    residimg = np.zeros(img.shape, img.dtype)

    #Use predictive coding to reduce the entropy of the image (input is single gray-scale
    #and output is a single greyscale image as output)
    #Each pixel is predicted as weighted average of adjacent 4 pixels in raster order
    #And calculate and encode the residual
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            prediction = predictor(r, c, img)
            actual = img[r][c]
            residual = actual - prediction  # prediction error
            residimg[r][c] = convert_enc_residual(residual)
    sp.imsave("%s-residual.tif" % filename.split("-")[0], residimg)


# range that residual will be in is [-p, 255-p] (just shifting [0, 255] over by p)
def convert_enc_residual(residual):
    r = residual % 256 # if residual == -3, r = 253 # store in section above 255-P and 255
    return r

def decode(residualsfile):
    rimg = sp.imread(residualsfile)
    rimg = rimg[:,:,1]
    origimg = np.zeros(rimg.shape, rimg.dtype)

    for r in range(rimg.shape[0]):
        for c in range(rimg.shape[1]):
            resid = rimg[r][c]
            prediction = predictor(r, c, origimg)
            residual = convert_dec_residual(resid, prediction)
            original = residual + prediction
            origimg[r][c] = original

    sp.imsave("%s-decoded.tif" % residualsfile.split("-")[0], origimg)

def convert_dec_residual(residual, prediction):
    if (255 - prediction) < residual:   # only occurs when residual was negative in the encoding
        residual = residual - 256
    return residual


def predictor(r, c, img):
    surrounding_pixels = None
    predicted = 0

    # 3 pixels above, 1 directly to left
    # Cover edge cases
    if r == 0:                      # Top row
        if c == 0:                  # First column of top row
            predicted = 0#img[r][c]
        else:
            predicted = img[r][c-1]
    else:                           # Any other row
        if c == 0:                  # First column of any other row (get above two pixels)
            surrounding_pixels = [img[r-1][c], img[r-1][c+1]]
        elif c == img.shape[1]-1:   # Last column of any other row (get three surrounding pixels)
            surrounding_pixels = [img[r][c-1], img[r-1][c-1], img[r-1][c]]
        else:                       # Any other pixel (get four pixels in raster position)
            surrounding_pixels = [img[r][c-1], img[r-1][c-1], img[r-1][c], img[r-1][c+1]]
        predicted = math.floor(np.average(surrounding_pixels))

    return predicted


def displayHist(filename):
    img = sp.imread(filename)
    img = img[:,:,1]
    pl.hist(img.flatten(), bins=256)
    pl.title('Histogram of %s' % filename)
    pl.xlabel('Intensity')
    pl.ylabel('Number of pixels')
    pl.show()


def main():
    written()
    programming()

if __name__ == "__main__":
    main()