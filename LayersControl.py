from PIL import Image, ImageDraw
from HybridAlgorithmImplementation import HybridAlgorithm
import json


class ImageLayersController:
    def __init__(self, current_image, lab=None, class_num=0, log_path='', layers_logged=False, layers_extracted=False):
        self.image = current_image
        self.lab = []
        self.class_num = 0
        self.layers_list = []
        self.operations = \
            ['get_layers', 'merge_layers', 'show_layers', 'return_all', 'work_with_layer', 'return_to_whole']

        # getting layers matrix & classes count:
        if lab is not None and layers_extracted:  # if data is passed to class, it's just initialized
            self.lab = lab
            self.class_num = class_num

        elif log_path != '' and layers_logged:  # if data is logged, it's uploading from JSON-file
            self.upload_preprocessed_layers(log_path=log_path)

        else:  # else start up hybrid algorithm
            ha = HybridAlgorithm(image_=image)
            self.lab, self.class_num = \
                ha.run_algorithm(log=False, log_path='logs/' + 'temp', save=False, save_path='stages/' + 'temp' + '/')

        self.normalize_layers()

    def upload_preprocessed_layers(self, log_path):
        self.class_num = 0
        with open(log_path + ".json", "r") as read_file:
            self.lab = json.load(read_file)
        for i in self.lab:
            for j in i:
                if j not in self.layers_list:
                    self.layers_list.append(j)
        self.class_num = len(self.layers_list)

    # normalization is needed so as classes in lab were ranged from 0 to class_count
    def normalize_layers(self):
        for y in range(len(self.lab)):
            for x in range(len(self.lab[0])):
                self.lab[y][x] = self.layers_list.index(self.lab[y][x])

    # get picture of layer by index
    def get_layer(self, layer_num):
        return self.image_layers[layer_num]

    # get all layers in pictures
    def show_different_layers(self):
        while True:
            i = int(input('Enter layers number:'))
            if i < 0:
                break
            else:
                self.temp_merge_few([i]).show()

    def get_layer_by_index(self, layer_num):
        return self.temp_merge_few([layer_num])

    def temp_merge_few(self, layers_list):
        merged = Image.new('RGB', self.image.size)
        draw = ImageDraw.Draw(merged)
        for y in range(len(self.lab)):
            for x in range(len(self.lab[0])):
                if self.lab[y][x] in layers_list:
                    draw.point(xy=(x, y), fill=(self.image.getpixel((x, y))))
        return merged

    def const_merge_few(self, layers_list):
        if len(layers_list) > 1:
            for y in range(len(self.lab)):
                for x in range(len(self.lab[0])):
                    if self.lab[y][x] in layers_list[1:]:
                        self.lab[y][x] = layers_list[0]

        return self.lab

    def return_all(self):
        return self.temp_merge_few(self.layers_list)

    def merge_pictures(self, layer_pic, full_pic, branch_list):
        new_pic = Image.new('RGB', self.image.size)
        draw = ImageDraw.Draw(new_pic)

        for y in range(len(self.lab)):
            for x in range(len(self.lab[0])):
                if self.lab[y][x] in branch_list:
                    draw.point(xy=(x, y), fill=(layer_pic.getpixel((x, y))))
                else:
                    draw.point(xy=(x, y), fill=(full_pic.getpixel((x, y))))
        return new_pic


if __name__ == '__main__':
    image = Image.open(r'images/vityan.jpg')
    ilc = ImageLayersController(current_image=image, log_path='logs/vityan', layers_logged=True)
    ilc.show_different_layers()
