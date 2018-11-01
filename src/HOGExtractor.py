# extracting HOG feature from image
# author: Ben Quick
# email: benquickdenn@foxmail.com
# date: 2018-10-29

import numpy as np
from skimage.feature import hog
from PIL import Image

SIZE = 256


# Convert RGB image to gray image
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray


# Extract the HOG feature in num-array format
def extract_hog_feature(image):
    image_used = image.resize((SIZE, SIZE), Image.ANTIALIAS)
    reshape_im = np.reshape(image_used, (SIZE, SIZE, 3))
    gray_im = rgb2gray(reshape_im) / 255.0

    fd = hog(gray_im, orientations=16, pixels_per_cell=[16, 16], cells_per_block=[2, 2], visualize=False,
             transform_sqrt=True, block_norm='L2-Hys')
    return fd




