# -*- coding: utf-8 -*-

import numpy
import math


from PIL import Image
numpy.set_printoptions(threshold=numpy.nan) # print all data

#################
# Image helpers #
#################

def save_image(arr, filename):
    """Takes a numpy array, in proper shape, and writes is an image
    Args:
        arr     : numpy array in shape (x, y, rgb)
        filename: name of the image to write
    """
    img = Image.fromarray(arr, "RGB")
    img.save(filename)


def image_to_array(file):
    """Read an image into a numpy 3d array
    Args:
        file: filepath to image
    Returns:
        A 3d numpy array of type uint8.
    """
    img = Image.open(file)
    # Convert the image to a numpy matrix of unsigned 8 bit integers.
    img_arr = numpy.asarray(img).astype(numpy.uint8)
    return img_arr


def flat_array_to_image(arr, dims, filename):
    """ Write a 2d array to an image
    Args:
        arr     : The numpy array that contains the data
        dims    : a triple containing the rows, columns and depth of the image respectively
        filename: filepath to write the imae to
    """
    reshaped = arr.reshape(dims)
    save_image(reshaped, filename)

##################
# Kernel Helpers #
##################

def normalize_kernel(kernel, dim):
    """Normalizes a kernel
    Args:
        kernel: a two-d kernel
    """
    for x in range(0, dim):
        for y in range(dim):
            kernel[x][y] = kernel[x][y] / numpy.sum(kernel)
    return kernel


def gaussian_kernel(dim, sigma):
    rows = dim
    cols = dim
    arr = numpy.empty([rows, cols]).astype(numpy.float32)
    center = dim / 2
    total = 0.0
    for x in range(0, rows):
        for y in range(0, cols):
            x_ = x - center
            y_ = y - center
            arr[x][y] = (1 / (2.0 * math.pi * math.pow(sigma, 2))) * math.pow(math.e, -1.0 * (
                (math.pow(x_, 2) + math.pow(y_, 2)) / (2.0 * math.pow(sigma, 2))))
            total = total + arr[x][y]

    return normalize_kernel(arr, dim)


def identity_kernel(dim):
    """ The identity kernel
    0 0 0
    0 1 0
    0 0 0
    """
    arr = numpy.empty([dim, dim]).astype(numpy.float32)
    arr.fill(0.0)
    arr[dim / 2][dim / 2] = 1.0
    return normalize(numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0, 2.0], [-1.0, 0.0, -1.0]]), 3)


def blur_kernel():
    """ Blurring kernel
    1 2 1
    2 4 2 * 1/16
    1 2 1
    """
    arr = (numpy.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])) * 1 / 16
    return normalize_kernel(arr, 3)


def apply_kernel(kernel, kernel_dim, height, width, img):
    """ Applies a kernel to an image
    Args:
       kernel    : the kernel array
       kernel_dim: the kernel width/height
       height    : the height of the image in px
       width     : the width of the image in px
       img       : the image array
    """
    kernel_mid = int(kernel_dim / 2)

    for img_r in range(kernel_mid, (height - kernel_mid)):
        for img_c in range(kernel_mid, (width - kernel_mid)):
            # Accumulators
            acc = numpy.array([0.0, 0.0, 0.0])
            # Apply the kernel
            for k_row in range(0, kernel_dim):
                for k_col in range(0, kernel_dim):
                    # Compute the pixel offset in the image
                    img_x = img_r + (k_row - kernel_mid)
                    img_y = img_c + (k_col - kernel_mid)
                    kernel_val = kernel[k_row][k_col]
                    img_val    = img[img_x][img_y]
                    acc += kernel_val * img_val
            # Set the new value on the pixel
            img[img_r][img_c] = acc
    return img


def apply_kernel_1d(kernel, kernel_dim, height, width, img):
    """ Applies a kernel to an image
    Args:
       kernel    : the kernel array
       kernel_dim: the kernel width/height
       height    : the height of the image in px
       width     : the width of the image in px
       img       : the image array
    """
    kernel_mid = int(kernel_dim / 2)
    img_out = numpy.copy(img)

    for img_r in range(kernel_mid, (height - kernel_mid)):
        for img_c in range(kernel_mid, (width - kernel_mid)):
            acc = 0.0
            for k_row in range(0, kernel_dim):
                for k_col in range(0, kernel_dim):

                    # X and Y of pixel on which kernel maps.
                    image_r_idx = img_r + (k_row - kernel_mid)
                    image_c_idx = img_c + (k_col - kernel_mid)
                    # Values from both.
                    kernel_value = kernel[k_row][k_col]
                    image_value  = img[image_r_idx * width + image_c_idx]

                    acc += kernel_value * image_value
            img_out[img_r * width + img_c] = acc
    return img_out