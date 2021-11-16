import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imageio
import math

img_url = "https://news.virginia.edu/sites/default/files/article_image/accolades_ss_header.jpg"
img = imageio.imread(img_url).astype('float32') / 255

# print('Image size: ', img.shape)


def displayImage(img):
    plt.imshow(img)
    plt.grid(False)
    plt.axis('off')
    plt.show()

def squareAdd(x, y):
    return x**2 + y**2


def g(x, y, sigma):
    frac = 1 / (2 * math.pi * (sigma**2))
    expo = (-1 * squareAdd(x, y)) / (2 * sigma**2)
    e = math.e**expo
    return frac*e


def gaussianMatrix(sigma, size):
    global kernel
    s2 = math.floor(size / 2)
    kernel = np.zeros(shape=(size, size))
    for x in range(0, size):
        for y in range(0, size):
            kernel[x][y] = g(s2-x, s2-y, sigma)
    kernel = kernel / kernel.sum()

gaussianMatrix(1, 5)
gaussianMatrix(4, 15)
gaussianMatrix(20, 81)
gaussianMatrix(40, 151)

def applyMatrix(img):
    global kernel
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    blurRed = signal.convolve2d(red, kernel, boundary='symm', mode='full')
    blurGreen = signal.convolve2d(green, kernel, boundary='symm', mode='full')
    blurBlue = signal.convolve2d(blue, kernel, boundary='symm', mode='full')

    blurredImage = np.dstack((blurRed, blurGreen, blurBlue))

    displayImage(blurredImage)


def blur(img, sigma, size):
    gaussianMatrix(sigma, size)
    applyMatrix(img)

displayImage(img)
blur(img, 4, 15)
blur(img, 10, 30)
blur(img, 20, 60)