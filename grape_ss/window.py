
import os
import numpy
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as compare_ssim

g_path = 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/test/bimage/'
gp_list = os.listdir(g_path)
grape_list = []

for name in gp_list:
    #grape_image = Image.open(g_path + name)
    #grape_image = numpy.array(grape_image)
    grape_image = cv2.imread(g_path + name)
    grape_list.append(grape_image)


def make_window(candi, path):  # input is min size of box
    input_image = cv2.imread(path)
    result_candi = []
    for x, y, w, h in candi:
        boxed_img = input_image[y: y + h, x: x + w]
        check = False
        for truth_img in grape_list:
            mini = min(boxed_img.shape[0], boxed_img.shape[1], truth_img.shape[0], truth_img.shape[1])
            boxed_img_r = cv2.resize(boxed_img, (mini, mini))
            truth_img_r = cv2.resize(truth_img, (mini, mini))
            boxed_img_r = cv2.cvtColor(boxed_img_r, cv2.COLOR_BGR2GRAY)
            truth_img_r = cv2.cvtColor(truth_img_r, cv2.COLOR_BGR2GRAY)
            score, _ = compare_ssim(boxed_img_r, truth_img_r, full=True)  # full=True: 이미지 전체에 대해서 구조 비교
            if score > 0.25:
                check = True
                break
        if check:
            result_candi.append([x, y, w, h])
    return result_candi


def make_window2(candi, path):  # input is min size of box
    input_image = cv2.imread(path)
    result_candi = []
    for x, y, w, h in candi:
        boxed_img = input_image[y: y + h, x: x + w]
        check = False
        for truth_img in grape_list:
            # maxi = max(boxed_img.shape[0], boxed_img.shape[1], truth_img.shape[0], truth_img.shape[1])
            boxed_img_r = cv2.resize(boxed_img, (256, 256))
            truth_img_r = cv2.resize(truth_img, (256, 256))
            boxed_img_r = cv2.cvtColor(boxed_img_r, cv2.COLOR_BGR2GRAY)
            truth_img_r = cv2.cvtColor(truth_img_r, cv2.COLOR_BGR2GRAY)
            score, _ = compare_ssim(boxed_img_r, truth_img_r, full=True)  # full=True: 이미지 전체에 대해서 구조 비교
            if score > 0.4:
                check = True
                break
        if check:
            result_candi.append([x, y, w, h])
    return result_candi

# zero padding 후 값 구해서 0.5 이상일 때만 detect → example
# numpy.pad(input, ((상, 하), (좌, 우)), 'constant', constant_value=0)


def ad_window(candi):
    result = []
    cent = []
    for box in candi:
        if box in result:
            continue
        if len(cent) == 0:
            result.append(box)
            cent.append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        else:
            sw = False
            for i in range(len(cent)):
                if abs(box[0]+box[2]/2-cent[i][0])<100 and abs(box[1]+box[3]/2-cent[i][1])<100:
                    sw = True
                if sw == False and i == len(cent) - 1:
                    result.append(box)
                    cent.append([box[0]+box[2]/2, box[1]+box[3]/2])
    return result
