from PIL import ImageFilter, ImageEnhance, Image
# import matplotlib as plt
import numpy as np
import os
import cv2
import random

class EditingMethods:
    def __init__(self):
        self.operations = ['blur', 'sharpen', 'brightness', 'contrast', 'color', 'sharpness', 'noise']

    # simple filters

    def blurred(self, image, times):
        blurred = image.filter(ImageFilter.BLUR)
        for i in range(times-1):
            blurred = blurred.filter(ImageFilter.BLUR)
        return blurred

    def sharped(self, image, times):
        sharped = image.filter(ImageFilter.SHARPEN)
        for i in range(times-1):
            sharped = sharped.filter(ImageFilter.SHARPEN)
        return sharped

    # PIL Image Enhancer

    def brightness(self, image, deg):
        if 0.0 <= deg <= 1.0:
            bright_enhancer = ImageEnhance.Brightness(image)
            bright_enhancer.enhance(deg)

    def contrast(self, image, deg):
        if 0.0 <= deg <= 1.0:
            contrast_enhancer = ImageEnhance.Contrast(image)
            contrast_enhancer.enhance(deg)

    def color(self, image, deg):
        if 0.0 <= deg <= 1.0:
            color_enhancer = ImageEnhance.Color(image)
            color_enhancer.enhance(deg)

    def sharpness(self, image, deg):
        if 0.0 <= deg <= 1.0:
            sharp_enhancer = ImageEnhance.Sharpness(image)
            sharp_enhancer.enhance(deg)

    # noise functions

    def noisy(self, image, noise_typ):
        image = np.array(image)
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return Image.fromarray(np.uint8(noisy))
        elif noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return Image.fromarray(np.uint8(out))

        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return Image.fromarray(np.uint8(noisy))

        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return Image.fromarray(np.uint8(noisy))

    def sp_noise(self, image_, prob):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        image = np.array(image_)
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]

        return Image.fromarray(np.uint8(output))