from ImageEditingControl import ImageEditingController
from PIL import Image
import os

if __name__ == '__main__':
    image_title = "vityan"
    dir_ = os.getcwd()
    image_path = os.path.join(dir_, 'images', image_title + '.jpg')
    log_path = os.path.join(dir_, 'logs/', image_title)

    image = Image.open(image_path)
    img_ctrl = ImageEditingController(image, log=True, log_path=log_path)
    img_ctrl.editing_process()
# aaaaaaa