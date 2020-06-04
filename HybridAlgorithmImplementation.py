from typing import List

import cv2
import os
from PIL import Image, ImageDraw
import math as m
from functools import cmp_to_key
import numpy as np
import json
import random


class HybridAlgorithm:
    def __init__(self, image_):
        self.image = image_
        self.lab, self.class_num = None, None

    def result_pic(self):
        lab_pic = Image.new('RGB', self.image.size)
        draw = ImageDraw.Draw(lab_pic)

        colors = {}
        for y in range(self.image.size[1]):
            for x in range(self.image.size[0]):
                # draw.point(xy=(x, y), fill=(10 * self.lab[y][x], 10 * self.lab[y][x], 10 * self.lab[y][x]))
                if self.lab[y][x] not in colors:
                    colors[self.lab[y][x]] = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
                draw.point(xy=(x, y),
                           fill=(colors[self.lab[y][x]][0], colors[self.lab[y][x]][1], colors[self.lab[y][x]][2]))
        return lab_pic

    def log_results(self, file_path):
        with open(file_path + ".json", "w") as write_file:
            json.dump(self.lab, write_file)

    def run_algorithm(self, log, log_path, save, save_path):
        alg_stages = []
        # print(self.image.size)
        ed = EdgeDetector(self.image)
        print("[INFO] performing holistically-nested edge detection...")
        self.image = ed.passing_through_net()
        if save:
            alg_stages.append(['hed', self.image])

        print("[INFO] performing image binarisation...")
        self.image = ed.turning_to_black_and_white()
        if save:
            alg_stages.append(['b_w', self.image])

        print("[INFO] performing forest fire method")
        cf = ClusterFinder(self.image)
        self.lab, self.class_num = cf.forest_fire_method()
        if save:
            alg_stages.append(['init_clusters', self.result_pic()])

        print("[INFO] deleting edges")
        self.lab, self.class_num = cf.edges_cutting_off()
        if save:
            alg_stages.append(['edges_deleted', self.result_pic()])

        print("[INFO] deleting redundant classes")
        cr = ClusterReducer(lab=self.lab, class_num=self.class_num)
        self.lab, self.class_num = cr.iterations('percentage')
        if save:
            alg_stages.append(['clusters_reduced', self.result_pic()])

        if log:
            print("[INFO] logging results")
            self.log_results(log_path)

        if save:
            print("[INFO] saving stages")
            for i in alg_stages:
                i[1].save(save_path + i[0] + ".jpg")

        return {'pic': self.result_pic(), 'matr': self.lab, 'num': self.class_num}

    def run_from_edged(self, log, log_path, save, save_path):
        alg_stages = []
        # print(self.image.size)
        ed = EdgeDetector(self.image)

        print("[INFO] performing image binarisation...")
        self.image = ed.turning_to_black_and_white()
        if save:
            alg_stages.append(['b_w', self.image])

        print("[INFO] performing forest fire method")
        cf = ClusterFinder(self.image)
        self.lab, self.class_num = cf.forest_fire_method()
        if save:
            alg_stages.append(['init_clusters', self.result_pic()])

        print("[INFO] deleting edges")
        self.lab, self.class_num = cf.edges_cutting_off()
        if save:
            alg_stages.append(['edges_deleted', self.result_pic()])

        print("[INFO] deleting redundant classes")
        cr = ClusterReducer(lab=self.lab, class_num=self.class_num)
        self.lab, self.class_num = cr.iterations('percentage')
        if save:
            alg_stages.append(['clusters_reduced', self.result_pic()])

        if log:
            print("[INFO] logging results")
            self.log_results(log_path)

        if save:
            print("[INFO] saving stages")
            for i in alg_stages:
                i[1].save(save_path + i[0] + ".jpg")

        return {'pic': self.result_pic(), 'matr': self.lab, 'num': self.class_num}


class EdgeDetector:
    def __init__(self, image_):
        # initializing hed model
        proto_path = os.path.sep.join(['hed_model', "deploy.prototxt"])
        model_path = os.path.sep.join(['hed_model', "hed_pretrained_bsds.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # register our new layer with the model
        cv2.dnn_registerLayer("Crop", CropLayer)

        # instances of cv2 and PIL images
        self.image, self.out_pil = image_, image_
        # image sizes
        self.H, self.W = None, None

    def open_image(self):
        # load the input image and grab its dimensions
        self.image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        (self.H, self.W) = self.image.shape[:2]

        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1.0, size=(self.W, self.H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        return blob

    def passing_through_net(self):
        self.net.setInput(self.open_image())
        hed = self.net.forward()
        hed = cv2.resize(hed[0, 0], (self.W, self.H))
        hed = (255 * hed).astype("uint8")

        hed = cv2.cvtColor(hed, cv2.COLOR_BGR2RGB)
        self.out_pil = Image.fromarray(hed)

        return self.out_pil

    def turning_to_black_and_white(self):
        thresh = 25
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
    def __init__(self, image_):
        self.image = image_
        self.W, self.H = image_.size

        # matrix of clusters
        self.lab = []
        self.class_count = 0

    def forest_fire_method(self):
        self.lab = [[0 for not_global_variable in range(self.W)] for another_not_global_variable in range(self.H)]
        for y in range(self.H):
            for x in range(self.W):
                if self.lab[y][x] == 0 and self.image.getpixel((x, y)) == 0:
                    self.class_count += 1
                    self.lab[y][x] = self.class_count
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
                                    self.lab[i][j] = self.class_count
                                    p_stack.append([i, j])

        return self.lab, self.class_count

    def edges_cutting_off(self):
        edges = [[-1 if j != 0 else 0 for j in i] for i in self.lab]
        while sum(i.count(0) for i in edges) > 0:
            for y in range(self.H):
                for x in range(self.W):
                    if edges[y][x] == 0:
                        nearest_classes = []
                        y_first = y if y == 0 else y - 1
                        y_last = y if y == self.H - 1 else y + 2
                        for i in range(y_first, y_last):
                            x_first = x if x == 0 else x - 1
                            x_last = x if x == self.W - 1 else x + 2
                            for j in range(x_first, x_last):
                                if edges[i][j] == -1:
                                    nearest_classes.append(self.lab[i][j])
                        try:
                            edges[y][x] = max(nearest_classes, key=lambda x_: nearest_classes.count(x_))
                        except ValueError:
                            pass

            self.lab = [[edges[i][j] if edges[i][j] not in [0, -1] else self.lab[i][j]
                         for j in range(len(self.lab[0]))] for i in range(len(self.lab))]
            edges = [[-1 if j != 0 else 0 for j in i] for i in self.lab]

        self.lab = [[j - 1 for j in i] for i in self.lab]
        return self.lab, self.class_count


class ClusterReducer:
    def __init__(self, lab, class_num):

        self.lab = lab
        self.class_num = class_num

        self.class_count: List[int] = self.count_classes()
        self.class_count_dict: List[List[int]] = self.make_classes_dict()

        self.centres_list = self.get_areas_centres()

    def count_classes(self):
        a = []
        for i in range(0, self.class_num):
            class_pixels_sum = 0
            for j in self.lab:
                class_pixels_sum += j.count(i)
            a.append(class_pixels_sum)
        return a

    def make_classes_dict(self):
        a = [[i, self.class_count[i]] for i in range(len(self.class_count))]
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
        for i in range(len(self.centres_list)):
            if self.centres_list[i][0] is not None:
                dist = self.find_distance(self.centres_list[i][1], current_center)
                if dist < min_dist and i != class_num and i != 0:
                    min_dist = dist
                    closest_class = i
        return closest_class

    def merge_closest_classes(self, class_num):
        closest_class_num = self.find_closest_class(class_num)
        for i in range(len(self.lab)):
            for j in range(len(self.lab[i])):
                if self.lab[i][j] == class_num:
                    self.lab[i][j] = closest_class_num
        self.centres_list[class_num] = [None, None]
        self.centres_list[closest_class_num] = \
            [closest_class_num, self.find_area_center(closest_class_num)]
        print('[INFO] classes', class_num, 'and', closest_class_num, 'merged!', sep=' ')
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

    def iterations(self, merging_type):
        i = 0
        if merging_type == 'by hand':
            ans_string = ''
            print('[INFO] Picture contains', len(self.class_count_dict), 'classes')
            while ans_string != 'N':
                while True:
                    try:
                        it = int(input('[REQ] Enter number of iterations:'))
                        break
                    except (ValueError, TypeError, NameError):
                        print('[INFO] Wrong input, try again.')

                for j in range(it):
                    self.full_merging(i)
                    i += 1

                ans_string = input('Enter `N` to finish merging')

        elif merging_type == 'percentage':
            print('[INFO] Reducing classes by percentage...')
            percentage = 0.03
            while self.class_count_dict[i][1] < int(round(len(self.lab) * len(self.lab[0]) * (percentage ** 2))):
                self.full_merging(i)
                i += 1

        return self.lab, len(self.class_count_dict) - i


if __name__ == '__main__':
    pic_name = "painting"
    dir_ = os.getcwd()
    path = os.path.join(dir_, "logs")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(dir_, "stages")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(dir_, "stages", pic_name)
    if not os.path.exists(path):
        os.mkdir(path)

    image = Image.open('images/' + pic_name + '.jpg')
    ha = HybridAlgorithm(image)
    ha.run_algorithm(log=True, log_path='logs/' + pic_name, save=True, save_path='stages/' + pic_name + '/')
    # ha.run_from_edged(log=True, log_path='logs/' + pic_name, save=True, save_path='stages/' + pic_name + '/')
