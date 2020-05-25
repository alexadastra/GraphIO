from ImageEditingControl import ImageEditingController
from PIL import Image

if __name__ == '__main__':

    image = Image.open(r'C:\Users\1\Desktop\смечные картинки\lina.jpg')
    img_ctrl = ImageEditingController(image, {})
    img_ctrl.editing_process()
