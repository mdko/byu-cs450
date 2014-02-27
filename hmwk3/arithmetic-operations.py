from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run_arithmetic_operations():
	# For some reason, when I used what I normally would
	# i.e. 	img = pl.imread("Frames/Cat0.pgm")
    #		pl.imshow(img)
    #		plt.show()
    # the image had all its colors inverted, so this
    # fixes it but is much more confusing...
    #
    # img = pili.open("Frames/Cat0.pgm")
    # img.load()
    # back = pili.new("RGB", img.size, (255, 255, 255))
    # back.paste(img)
    # pl.imshow(np.asarray(back)) # PIL and matplotlib don't use same conventions, else would flip
    # plt.show()

    # results in image flipped, but as long as I'm consistent
    shape = plt.imread("Frames/Cat0.pgm").shape
    reduce_noise(2, shape)
    reduce_noise(5, shape)
    reduce_noise(10, shape)
    reduce_noise(20, shape)
    reduce_noise(40, shape)


def reduce_noise(num_images, shape):
    new_img = np.zeros(shape)
    for i in range(1, num_images):
        next_img_name = "Frames/Cat" + str(i) + ".pgm"
        next_img = plt.imread(next_img_name)
        new_img += next_img

    new_img = new_img * (1.0/num_images)    

    # pl.imshow(np.asarray(new_img), cmap=cm.Greys_r)
    new_img_name = "Cat_with_" + str(num_images) + ".png"
    plt.imsave(new_img_name, new_img, cmap=cm.Greys_r)


def run_spatial_filtering():
    # img_name = "noisygull"
    # img = np.asarray(plt.imread(img_name + ".png")[:,:,0])
    # uniform_averaging(img, img_name, 3)
    # uniform_averaging(img, img_name, 5)
    # uniform_averaging(img, img_name, 7)
    # uniform_averaging(img, img_name, 9)

    # median_filtering(img, img_name, 3)
    # median_filtering(img, img_name, 5)
    # median_filtering(img, img_name, 7)

    # img_name = "whitebox"
    # img = np.asarray(plt.imread(img_name + ".png")[:,:,0])
    # uniform_averaging(img, img_name, 3)
    # uniform_averaging(img, img_name, 5)
    # uniform_averaging(img, img_name, 7)
    # uniform_averaging(img, img_name, 9)

    #sobel_kernel(img, img_name) # does both x and y
    #laplacian_kernel(img, img_name)

    # gradient_magnitude_image(img, img_name)

    img_name = "blocks"
    img = np.asarray(plt.imread(img_name + ".png")[:,:,0])
    # gradient_magnitude_image(img, img_name)
    # blurred_img_3x3 = uniform_averaging(img, img_name, 3)
    # gradient_magnitude_image(blurred_img_3x3, img_name + "_blurred_3x3")

    # blurred_img_5x5 = uniform_averaging(img, img_name, 5)
    # gradient_magnitude_image(blurred_img_5x5, img_name + "_blurred_5x5")

    # blurred_img_7x7 = uniform_averaging(img, img_name, 7)
    # gradient_magnitude_image(blurred_img_7x7, img_name + "_blurred_7x7")

    # blurred_img_9x9 = uniform_averaging(img, img_name, 9)
    # gradient_magnitude_image(blurred_img_9x9, img_name + "_blurred_9x9")

    # Add to A(original_strength) the amount 1's, which depends on the radius
    unsharp_masking(img, img_name, orig_strength=2, radius=3)
    unsharp_masking(img, img_name, orig_strength=3, radius=3)
    unsharp_masking(img, img_name, orig_strength=4, radius=3)
    unsharp_masking(img, img_name, orig_strength=5, radius=3)
    unsharp_masking(img, img_name, orig_strength=6, radius=3)
    # unsharp_masking(img, img_name, orig_strength=2, radius=5)
    # unsharp_masking(img, img_name, orig_strength=3, radius=5)
    # unsharp_masking(img, img_name, orig_strength=4, radius=5)
    # unsharp_masking(img, img_name, orig_strength=5, radius=5)
    # unsharp_masking(img, img_name, orig_strength=6, radius=5)
    unsharp_masking(img, img_name, orig_strength=2, radius=7)
    unsharp_masking(img, img_name, orig_strength=3, radius=7)
    unsharp_masking(img, img_name, orig_strength=4, radius=7)
    unsharp_masking(img, img_name, orig_strength=5, radius=7)
    unsharp_masking(img, img_name, orig_strength=6, radius=7)


def spatial_filtering(img, kernel, spat_func, normalization=None):
    new_img = np.zeros((img.shape[0], img.shape[1]), img.dtype) # I am only caring about one 2 dimensions here, to make life easier
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
                new_img[r, c] = spat_func(img, kernel, r, c, normalization)

    return new_img


def unsharp_masking(img, img_name, orig_strength=1, radius=3):
    kernel = fill_unsharp_mask_kernel(orig_strength, radius)
    new_img = spatial_filtering(img, kernel, calculate_normal, 1.0)
    new_img = trim_extremes(new_img)
    new_img_name = img_name + "_unsharp_masking_s(" + str(orig_strength) + ")-r(" + str(radius) +").png"
    plt.imsave(new_img_name, new_img, cmap=cm.Greys_r)
    return new_img


def trim_extremes(img):
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            if (img[r, c] < 0.0):
                img[r, c] = 0.0
            elif (img[r, c] > 1.0):
                img[r, c] = 1.0
    return img


def fill_unsharp_mask_kernel(original_strength, radius):
    kernel = np.zeros((radius, radius))
    col_row_with_ones = (radius - 1)/2
    kernel[:,col_row_with_ones] = -1
    kernel[col_row_with_ones,:] = -1
    kernel[col_row_with_ones, col_row_with_ones] = original_strength + (col_row_with_ones * 4)
    kernel = kernel * (1.0/original_strength)
    return kernel


def gradient_magnitude_image(img, img_name):
    new_img = np.zeros(img.shape, img.dtype)
    sobel_x, sobel_y = sobel_kernel(img, img_name)
    new_img_name = img_name + "_gradient_magnitude.png"
    new_img = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    plt.imsave(new_img_name, new_img, cmap=cm.Greys_r)
    return new_img


def uniform_averaging(img, img_name, dim):
    kernel = np.ones((dim, dim))
    new_img_name = img_name + "_" + str(dim) + "x" + str(dim) + ".png"
    normalizing_factor = kernel.shape[0] * kernel.shape[1]
    new_img = spatial_filtering(img, kernel, calculate_normal, normalizing_factor)
    plt.imsave(new_img_name, new_img, cmap=cm.Greys_r)
    return new_img


def median_filtering(img, img_name, dim):
    kernel = np.ones((dim, dim))
    new_img_name = img_name + "_" + str(dim) + "x" + str(dim) + ".png"
    new_img = spatial_filtering(img, kernel, calculate_median)
    plt.imsave(new_img_name, new_img, cmap=cm.Greys_r)
    return new_img

def sobel_kernel(img, img_name):
    x_kernel = np.zeros((3, 3))
    y_kernel = np.zeros((3, 3))

    x_kernel[0, 0] = -1
    x_kernel[0, 2] = 1
    x_kernel[1, 0] = -2
    x_kernel[1, 2] = 2
    x_kernel[2, 0] = -1
    x_kernel[2, 2] = 1

    y_kernel = x_kernel.transpose()

    x_new_img_name = img_name + "_sobel_kernel_x.png"
    y_new_img_name = img_name + "_sobel_kernel_y.png"
    normalizing_factor = 8.0
    x_kernel_img = spatial_filtering(img, x_kernel, calculate_normal, normalizing_factor)
    y_kernel_img = spatial_filtering(img, y_kernel, calculate_normal, normalizing_factor)

    plt.imsave(x_new_img_name, x_kernel_img, cmap=cm.Greys_r)
    plt.imsave(y_new_img_name, y_kernel_img, cmap=cm.Greys_r)

    return x_kernel_img, y_kernel_img


def laplacian_kernel(img, img_name):
    laplacian_kernel = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    new_img_name = img_name + "_laplacian.png"
    new_img = spatial_filtering(img, laplacian_kernel, calculate_normal, 1.0)
    plt.imsave(new_img_name, new_img, cmap=cm.Greys_r)
    return new_img


def calculate_normal(img, kernel, row, col, normalizing_factor):
    avg = 0
    a = int((kernel.shape[0] - 1) / 2)
    b = int((kernel.shape[1] - 1) / 2)
    for s in range(-a, a+1):
        for t in range(-b, b+1):
            if (row+s < 0) or (row+s >= img.shape[0]) \
                or (col+t < 0) or (col+t >= img.shape[1]):
                avg += 0
            else:
                avg += img[row + s, col + t] * kernel[a + s, b + t]
    return avg / float(normalizing_factor)


def calculate_median(img, kernel, row, col, normalizing_factor=None):
    arr = np.ndarray((1,1), img.dtype)
    a = int((kernel.shape[0] - 1) / 2)
    b = int((kernel.shape[1] - 1) / 2)
    for s in range(-a, a+1):
        for t in range(-b, b+1):
            if (row+s < 0) or (row+s >= img.shape[0]) \
                or (col+t < 0) or (col+t >= img.shape[1]):
                arr = np.append(arr, 0)
            else:
                arr = np.append(arr, img[row + s, col + t] * kernel[a + s, b + t])

    return np.median(arr)



def main():
    #run_arithmetic_operations()
    run_spatial_filtering()


# boiler-plate code
if __name__ == "__main__":
    main()