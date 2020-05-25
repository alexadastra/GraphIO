from PIL import ImageFilter, ImageEnhance
import matplotlib as plt


class EditingMethods:
    def __init__(self):
        self.operations = ['blur', 'sharpen']

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
