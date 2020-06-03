from typing import List

import cv2
import os
from PIL import Image, ImageDraw
import math as m
from functools import cmp_to_key
import numpy as np
import json


class HybridAlgorithm:
    def __init__(self, image):
        self.image = image
        self.lab, self.class_num = None, None

    def result_pic(self):
        lab_pic = Image.new('RGB', self.image.size)
        draw = ImageDraw.Draw(lab_pic)

        for y in range(self.image.size[1]):
            for x in range(self.image.size[0]):
                draw.point(xy=(x, y), fill=(10 * self.lab[y][x], 10 * self.lab[y][x], 10 * self.lab[y][x]))

        return lab_pic

    def log_results(self, file_name):
        f = open(r'logs/' + file_name + '.txt', 'w')
        for i in self.lab:
            for j in i:
                f.write(str(j))
            f.write('\n')

        with open("logs/data_file.json", "w") as write_file:
            json.dump(self.lab, write_file)

    def run_algorithm(self):
        ed = EdgeDetector()
        self.image = ed.passing_through_net(image=self.image)
        self.image = ed.turning_to_black_and_white()

        cf = ClusterFinder(self.image)
        self.lab, self.class_num = cf.forest_fire_method()

        self.result_pic().show()

        cr = ClusterReducer(lab=self.lab, class_num=self.class_num)
        self.lab, self.class_num = cr.iterations()

        self.result_pic().show()

        return {'pic': self.result_pic(), 'matr': self.lab, 'num': self.class_num}


class EdgeDetector:
    def __init__(self):
        # initializing hed model
        proto_path = os.path.sep.join(['hed_model', "deploy.prototxt"])
        model_path = os.path.sep.join(['hed_model', "hed_pretrained_bsds.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # register our new layer with the model
        cv2.dnn_registerLayer("Crop", CropLayer)

        # instances of cv2 and PIL images
        self.image, self.out_pil = None, None
        # image sizes
        self.H, self.W = None, None

    def open_image(self, image):
        # load the input image and grab its dimensions
        self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        (self.H, self.W) = self.image.shape[:2]

        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1.0, size=(self.W, self.H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        return blob

    def passing_through_net(self, image):
        print("[INFO] performing holistically-nested edge detection...")
        self.net.setInput(self.open_image(image))
        hed = self.net.forward()
        hed = cv2.resize(hed[0, 0], (self.W, self.H))
        hed = (255 * hed).astype("uint8")

        hed = cv2.cvtColor(hed, cv2.COLOR_BGR2RGB)
        self.out_pil = Image.fromarray(hed)

        return self.out_pil

    def turning_to_black_and_white(self):
        thresh = 50
        self.out_pil = self.out_pil.convert('L').point(lambda x: 255 if x > thresh else 0, mode='1')

        return self.out_pil


class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


class ClusterFinder:
    def __init__(self, image):
        self.image = image
        self.W, self.H = image.size

        # matrix of clusters
        self.lab = []

    def forest_fire_method(self):
        self.lab = [[0 for not_global_variable in range(self.W)] for another_not_global_variable in range(self.H)]
        RegLS = 0
        for y in range(self.H):
            for x in range(self.W):
                if self.lab[y][x] == 0 and self.image.getpixel((x, y)) == 0:
                    RegLS += 1
                    self.lab[y][x] = RegLS
                    p_stack = [[y, x]]
                    while len(p_stack) > 0:
                        last = p_stack.pop()

                        y_first = last[0] if last[0] == 0 else last[0] - 1
                        y_last = last[0] if last[0] == self.H - 1 else last[0] + 2
                        for i in range(y_first, y_last):
                            x_first = last[1] if last[1] == 0 else last[1] - 1
                            x_last = last[1] if last[1] == self.W - 1 else last[1] + 2
                            for j in range(x_first, x_last):
                                if self.lab[i][j] == 0 and self.image.getpixel((j, i)) == 0:
                                    self.lab[i][j] = RegLS
                                    p_stack.append([i, j])

        return self.lab, RegLS


class ClusterReducer:
    def __init__(self, lab, class_num):

        self.lab = lab
        self.class_num = class_num

        self.class_count: List[int] = self.count_classes()
        self.class_count_dict: List[List[int]] = self.make_classes_dict()

        self.centres_list = self.get_areas_centres()

    def count_classes(self):
        a = []
        for i in range(1, self.class_num + 1):
            class_pixels_sum = 0
            for j in self.lab:
                class_pixels_sum += j.count(i)
            a.append(class_pixels_sum)
        return a

    def make_classes_dict(self):
        a = [[i, self.class_count[i]] for i in range(1, len(self.class_count))]
        a.sort(key=lambda x: x[1])
        return a

    def find_area_center(self, class_num):
        sum_x, sum_y, dot_num = 0, 0, 0
        for i in range(len(self.lab)):
            for j in range(len(self.lab[i])):
                if self.lab[i][j] == class_num:
                    sum_x += j
                    sum_y += i
                    dot_num += 1
        return [sum_x // dot_num, sum_y // dot_num]

    def get_areas_centres(self):
        a = []
        for i in range(len(self.class_count)):
            a.append([i, self.find_area_center(i)])
        return a

    @staticmethod
    def find_distance(point1, point2):
        return m.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def find_closest_class(self, class_num):
        min_dist = self.find_distance([0, 0], [len(self.lab), len(self.lab[0])])
        closest_class = -1
        current_center = self.centres_list[class_num][1]
        print(current_center)
        for i in range(len(self.centres_list)):
            if self.centres_list[i][0] is not None:
                dist = self.find_distance(self.centres_list[i][1], current_center)
                if dist < min_dist and i != class_num and i != 0:
                    min_dist = dist
                    closest_class = i
        return closest_class

    def merge_closest_classes(self, class_num):
        print(class_num)
        closest_class_num = self.find_closest_class(class_num)
        for i in range(len(self.lab)):
            for j in range(len(self.lab[i])):
                if self.lab[i][j] == class_num:
                    self.lab[i][j] = closest_class_num
        self.centres_list[class_num] = [None, None]
        self.centres_list[closest_class_num] = \
            [closest_class_num, self.find_area_center(closest_class_num)]
        print('classes', class_num, 'and', closest_class_num, 'merged!', sep=' ')
        return closest_class_num

    @staticmethod
    def compare(x, y):
        if x[1] is None and y[1] is None:
            return 1
        if x[1] is None or y[1] is None:
            return 1
        return x[1] - y[1]

    def full_merging(self, arg):
        smallest_class = self.class_count_dict[arg][0]
        closest_class_num = self.merge_closest_classes(smallest_class)
        previous_value = self.class_count_dict[arg][1]
        self.class_count_dict[arg] = [None, None]
        for i in self.class_count_dict:
            if i[0] == closest_class_num:
                i = [closest_class_num, i[1] + previous_value]
                break
        self.class_count_dict.sort(key=cmp_to_key(self.compare))

    def iterations(self):
        i = 0
        ans_string = ''
        print('Picture contains', len(self.class_count_dict), 'classes')
        while ans_string != 'N':
            it = int(input('Enter number of iterations:'))
            for j in range(it):
                self.full_merging(i)
                i += 1

            # lab_pic_class = Image.new('RGB', out_pil.size)
            # draw_class = ImageDraw.Draw(lab_pic_class)
            # for y in range(out_pil.size[1]):
            #     for x in range(out_pil.size[0]):
            #         draw_class.point(xy=(x, y), fill=(20 * lab[y][x], 20 * lab[y][x], 20 * lab[y][x]))
            #
            # lab_pic_class.show()

            ans_string = input('Enter `N` to finish merging')

        return self.lab, len(self.class_count_dict) - i


if __name__ == '__main__':
    image = Image.open('images/vityan.jpg')
    ha = HybridAlgorithm(image)
    ha.run_algorithm()
    ha.log_results('vityan')
