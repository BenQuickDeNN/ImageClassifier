# extracting color feature from image
# author: Ben Quick
# email: benquickdenn@foxmail.com
# date: 2018-10-31

import numpy as np
from PIL import Image

SIZE = 256


def extract_color_feature(image):
    pixels_red = []
    pixels_green = []
    pixels_blue = []
    image_height = image.size[0]
    image_width = image.size[1]
    for i in range(0, image_height):
        for j in range(0, image_width):
            pixel_color = image.getpixel((i, j))
            pixels_red.append(pixel_color[0])
            pixels_green.append(pixel_color[1])
            pixels_blue.append(pixel_color[2])
    pixel_red_average = np.average(pixels_red)
    pixel_green_average = np.average(pixels_green)
    pixel_blue_average = np.average(pixels_blue)
    pixel_color_average = [pixel_red_average, pixel_green_average, pixel_blue_average]
    return pixel_color_average


# test region
# extract_color_feature('../training-2000/images/MWI_00Y7L33j7KcV6fTv.jpg')
