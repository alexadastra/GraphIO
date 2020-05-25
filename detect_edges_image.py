# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import argparse
from typing import List

import cv2
import os
from PIL import Image, ImageDraw
import math as m
from functools import cmp_to_key
import copy

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--edge-detector", type=str, required=True,
#     help="path to OpenCV's deep learning edge detector")
# ap.add_argument("-i", "--image", type=str, required=True,
#     help="path to input image")
# args = vars(ap.parse_args())


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


def predict(image_path):
    # load our serialized edge detector from disk
    print("[INFO] loading edge detector...")
    # protoPath = os.path.sep.join([args["edge_detector"],
    #     "deploy.prototxt"])
    # modelPath = os.path.sep.join([args["edge_detector"],
    #     "hed_pretrained_bsds.caffemodel"])

    protoPath = os.path.sep.join(['hed_model', "deploy.prototxt"])
    modelPath = os.path.sep.join(['hed_model', "hed_pretrained_bsds.caffemodel"])

    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # register our new layer with the model
    cv2.dnn_registerLayer("Crop", CropLayer)

    # load the input image and grab its dimensions
    # image = cv2.imread(args["image"])
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # convert the image to grayscale, blur it, and perform Canny
    # edge detection
    # print("[INFO] performing Canny edge detection...")
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny = cv2.Canny(blurred, 30, 150)

    # construct a blob out of the input image for the Holistically-Nested
    # Edge Detector
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)

    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    # show the output edge detection results for Canny and
    # Holistically-Nested Edge Detection
    # cv2.imshow("Input", image)
    # cv2.imshow("Canny", canny)
    # cv2.imshow("HED", hed)
    # cv2.waitKey(0)

    # canny = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
    # out_pil = Image.fromarray(canny)
    # out_pil.show()

    hed = cv2.cvtColor(hed, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(hed)
    return out_pil


out_pil = predict('images/liza.jpg')
out_pil.show()

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# in_pil = Image.fromarray(image)
# in_pil.show()
print("[INFO] Turning image to black and white")

thresh = 50
fn = lambda x: 255 if x > thresh else 0
out_pil = out_pil.convert('L').point(fn, mode='1')
out_pil.show()

lab_pic = Image.new('RGB', out_pil.size)
draw = ImageDraw.Draw(lab_pic)

lab = [[0 for j in range(out_pil.size[0])] for i in range(out_pil.size[1])]
RegLS = 0
for y in range(out_pil.size[1]):
    for x in range(out_pil.size[0]):
        if lab[y][x] == 0 and out_pil.getpixel((x, y)) == 0:
            RegLS += 1
            lab[y][x] = RegLS
            p_stack = [[y, x]]
            while len(p_stack) > 0:
                last = p_stack.pop()

                y_first = last[0] if last[0] == 0 else last[0] - 1
                y_last = last[0] if last[0] == out_pil.size[1] - 1 else last[0] + 2
                for i in range(y_first, y_last):
                    x_first = last[1] if last[1] == 0 else last[1] - 1
                    x_last = last[1] if last[1] == out_pil.size[0] - 1 else last[1] + 2
                    for j in range(x_first, x_last):
                        # try:
                        if lab[i][j] == 0 and out_pil.getpixel((j, i)) == 0:
                            lab[i][j] = RegLS
                            p_stack.append([i, j])

print("[INFO] Showing results")

for y in range(out_pil.size[1]):
    for x in range(out_pil.size[0]):
        draw.point(xy=(x, y), fill=(10 * lab[y][x], 10 * lab[y][x], 10 * lab[y][x]))

lab_pic.show()

print("[INFO] Logging results")

f = open(r'logs\log.txt', 'w')
for i in lab:
    for j in i:
        f.write(str(j))
    f.write('\n')

print("[INFO] Counting items is classes")

class_count = []
for i in range(1, RegLS + 1):
    sum = 0
    for j in lab:
        sum += j.count(i)
    class_count.append(sum)

print("[INFO] Turning to dict")

class_count_dict: List[List[int]] = [[i, class_count[i]] for i in range(1, len(class_count))]
class_count_dict.sort(key=lambda x: x[1])

print('first class is:', class_count_dict[0])


def find_area_center(class_num):
    sum_x, sum_y, dot_num = 0, 0, 0
    for i in range(len(lab)):
        for j in range(len(lab[i])):
            if lab[i][j] == class_num:
                sum_x += j
                sum_y += i
                dot_num += 1
    return [sum_x // dot_num, sum_y // dot_num]


def get_areas_centres():
    a = []
    for i in range(len(class_count)):
        a.append([i, find_area_center(i)])
    return a


def find_distance(point1, point2):
    return m.sqrt((point1[0]-point2[0]) ** 2 + (point1[1]-point2[1]) ** 2)


def find_closest_class(class_num):
    min_dist = find_distance([0, 0], [out_pil.size[0], out_pil.size[1]])
    closest_class = -1
    current_center = centres_list[class_num][1]
    print(current_center)
    for i in range(len(centres_list)):
        if centres_list[i][0] != None:
            dist = find_distance(centres_list[i][1], current_center)
            if dist < min_dist and i != class_num:
                min_dist = dist
                closest_class = i
            # print(i)
    return closest_class


def merge_closest_classes(class_num):
    print(class_num)
    closest_class_num = find_closest_class(class_num)
    for i in range(len(lab)):
        for j in range(len(lab[i])):
            if lab[i][j] == class_num:
                lab[i][j] = closest_class_num
    centres_list[class_num] = [None, None]
    centres_list[closest_class_num] = \
    [closest_class_num, find_area_center(closest_class_num)]
    print('classes', class_num, 'and', closest_class_num, 'merged!', sep=' ')
    return closest_class_num


print('[INFO] Counting classes centres')

centres_list = get_areas_centres()

print('[INFO] Merging minor classes')


def compare(x, y):
    if x[1] == None and y[1] == None:
        return 1
    if x[1] == None or y[1] == None:
        return 1
    return x[1] - y[1]


def full_merging(arg):
    smallest_class = class_count_dict[arg][0]
    closest_class_num = merge_closest_classes(smallest_class)
    previous_value = class_count_dict[arg][1]
    class_count_dict[arg] = [None, None]
    for i in class_count_dict:
        if i[0] == closest_class_num:
            i = [closest_class_num, i[1] + previous_value]
            break
    class_count_dict.sort(key=cmp_to_key(compare))


def iterations():
    i = 0
    ans_string = ''
    print('Picture contains', len(class_count_dict), 'classes')
    while ans_string != 'N':
        it = int(input('Enter number of iterations:'))
        for j in range(it):
            full_merging(i)
            i += 1

        lab_pic_class = Image.new('RGB', out_pil.size)
        draw_class = ImageDraw.Draw(lab_pic_class)
        for y in range(out_pil.size[1]):
            for x in range(out_pil.size[0]):
                draw_class.point(xy=(x, y), fill=(20 * lab[y][x], 20 * lab[y][x], 20 * lab[y][x]))

        lab_pic_class.show()

        ans_string = input('Enter `N` to finish merging')


iterations()
