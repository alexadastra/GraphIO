from PIL import Image, ImageDraw
from HybridAlgorithmImplementation import HybridAlgorithm


class ImageLayersController:
    def __init__(self, current_image, lab=None, class_num=0, log_path='', layers_logged=False, layers_extracted=False):
        self.image = current_image
        self.lab = []
        self.class_num = 0

        if lab is not None and layers_extracted:
            self.lab = lab
            self.class_num = class_num

        elif log_path != '' and layers_logged:
            self.upload_preprocessed_layers(log_path=log_path)

        else:
            ha = HybridAlgorithm(image=image)
            self.lab, self.class_num = ha.run_algorithm()

        self.image_layers = [Image.new('RGB', self.image.size)] * self.class_num
        self.structure_layers()

    def upload_preprocessed_layers(self, log_path):
        f = open(log_path, 'r')
        self.class_num = 0
        for i in f:
            a = []
            for j in i:
                if j != '\n':
                    j = int(j)
                    a.append(j)
                    if j > self.class_num:
                        self.class_num = j + 1
            self.lab.append(a)

    def structure_layers(self):

        image_draws = [ImageDraw.Draw(i) for i in self.image_layers]
        for y in range(len(self.lab)):
            for x in range(len(self.lab[0])):
                idx = self.lab[y][x]
                image_draws[idx].point(xy=(x, y), fill=(self.image.getpixel((x, y))))

    def get_layer(self, layer_num):
        return self.image_layers[layer_num]

    def show_different_layers(self):
        while True:
            i = int(input('Enter layers number:'))
            if i < 0:
                break
            else:
                self.get_layer(i).show()


if __name__ == '__main__':
    image = Image.open(r'images/vityan.jpg')
    ilc = ImageLayersController(current_image=image, log_path='logs/vityan.txt', layers_logged=True)
    ilc.show_different_layers()
