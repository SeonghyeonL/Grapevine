from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
# import customsearch
from PIL import Image
import numpy as np
import window
import os
import compute

# nums = '0490'  # 486
i_path = 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/image/'
i_list = os.listdir(i_path)
for i in range(len(i_list)):
    i_list[i] = i_list[i][:-4]

avg_iou = 0
avg_nbr = 0

for nums in i_list:

    # loading astronaut image
    #img = skimage.data.astronaut()
    #img = Image.open('C:/Users/kate1/Desktop/eedesign/wgisd-master-data/data/CDY_2015.jpg')
    img = Image.open('C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/image/'+nums+'.jpg')
    img2 = img.resize((256, 256))
    img3 = img.resize((128, 128))
    img = np.array(img)
    img2 = np.array(img2)
    img3 = np.array(img3)

    train_label = np.loadtxt('C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/label/'+nums+'.txt', dtype='float')
    # train_label = np.loadtxt('C:/Users/kate1/Desktop/eedesign/wgisd-master-data/data/CDY_2015.txt', dtype='float')

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=200, sigma=0.9, min_size=10)
    img_lbl2, regions2 = selectivesearch.selective_search(
        img2, scale=150, sigma=0.9, min_size=10)
    img_lbl3, regions3 = selectivesearch.selective_search(
        img3, scale=100, sigma=0.9, min_size=10)

    # make a list of regions
    candi = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candi:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:  # minimum size = 2000
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:  # except long rect
            continue
        candi.add(r['rect'])
    candi2 = set()
    for r in regions2:
        if r['rect'] in candi2:
            continue
        if r['size'] < 2000/4:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:  # except long rect
            continue
        candi2.add(r['rect'])
    candi3 = set()
    for r in regions3:
        if r['rect'] in candi2:
            continue
        if r['size'] < 2000/16:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:  # except long rect
            continue
        candi3.add(r['rect'])

    # for window input
    candis = set()
    for r in regions:
        if r['size'] < 9000:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candis.add((x, y, w, h))
    for r in regions2:
        if r['size'] < 9000/4:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:  # except long rect
            continue
        candis.add((x*2, y*2, w*2, h*2))
    for r in regions3:
        if r['size'] < 9000/16:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:  # except long rect
            continue
        candis.add((x*4, y*4, w*4, h*4))

    candis1 = window.make_window(candis, 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/image/' + nums + '.jpg')
    candis2 = window.make_window2(candis, 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/image/' + nums + '.jpg')
    candis_s = []
    for r in candis1:
        if r in candis2:
            candis_s.append(r)

    result = window.ad_window(candis_s)

    # for example image
    if nums == '0486' or nums == '0490':
        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(6, 6))
        for xi in range(2):
            for yi in range(4):
                ax[xi, yi].imshow(img)
        for x, y, w, h in train_label:  # r['rect']
            rect = mpatches.Rectangle(  # (x, y), w, h
                (x, y), w, h, fill=False, edgecolor='blue', linewidth=1)
            # (x*2048-w*2048*0.5, y*1365-h*1365*0.5), w*2048, h*1365
            ax[0, 0].add_patch(rect)
        for x, y, w, h in candi:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax[0, 1].add_patch(rect)
        for x, y, w, h in candi2:
            rect = mpatches.Rectangle(
                (x * 2, y * 2), w * 2, h * 2, fill=False, edgecolor='red', linewidth=1)
            ax[0, 2].add_patch(rect)
        for x, y, w, h in candi3:
            rect = mpatches.Rectangle(
                (x * 4, y * 4), w * 4, h * 4, fill=False, edgecolor='red', linewidth=1)
            ax[0, 3].add_patch(rect)
        for x, y, w, h in candis1:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax[1, 0].add_patch(rect)
        for x, y, w, h in candis2:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax[1, 1].add_patch(rect)
        for x, y, w, h in candis_s:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax[1, 2].add_patch(rect)
        for x, y, w, h in result:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax[1, 3].add_patch(rect)
        plt.savefig(nums+'_result.png')

    iou = compute.intersection_over_union(result, train_label)
    nbr = abs(len(result)-len(train_label)) / len(train_label)
    print(nums, iou, nbr)
    avg_iou += iou
    avg_nbr += nbr

avg_iou /= len(i_list)
avg_nbr /= len(i_list)
print("average iou:", avg_iou)
print("average nbr:", avg_nbr)

'''
fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(6, 6))  # fig = plt.figure() + ax = fig.add_subplot(111)
ax[0, 0].imshow(img)  # show
for x, y, w, h in train_label:  # r['rect']
    rect = mpatches.Rectangle(  # (x, y), w, h
        (x, y), w, h, fill=False, edgecolor='blue', linewidth=1)
    # (x*2048-w*2048*0.5, y*1365-h*1365*0.5), w*2048, h*1365
    ax[0, 0].add_patch(rect)
ax[0, 1].imshow(img)
for x, y, w, h in candi:
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax[0, 1].add_patch(rect)
ax[0, 2].imshow(img)
for x, y, w, h in candi2:
    rect = mpatches.Rectangle(
        (x*2, y*2), w*2, h*2, fill=False, edgecolor='red', linewidth=1)
    ax[0, 2].add_patch(rect)
ax[0, 3].imshow(img)
for x, y, w, h in candi3:
    rect = mpatches.Rectangle(
        (x*4, y*4), w*4, h*4, fill=False, edgecolor='red', linewidth=1)
    ax[0, 3].add_patch(rect)
ax[1, 0].imshow(img)
candis1 = window.make_window(candis, 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/image/'+nums+'.jpg')
for x, y, w, h in candis1:
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax[1, 0].add_patch(rect)
ax[1, 1].imshow(img)
candis2 = window.make_window2(candis, 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/train/image/'+nums+'.jpg')
for x, y, w, h in candis2:
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax[1, 1].add_patch(rect)
ax[1, 2].imshow(img)
candis_s = []
for r in candis1:
    if r in candis2:
        candis_s.append(r)
for x, y, w, h in candis_s:
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax[1, 2].add_patch(rect)
ax[1, 3].imshow(img)
for x, y, w, h in window.ad_window(candis_s):
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax[1, 3].add_patch(rect)
plt.show()
'''
