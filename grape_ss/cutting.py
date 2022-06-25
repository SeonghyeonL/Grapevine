
import numpy as np
from PIL import Image
# import cv2
import random
import glob  # img_list = glob.glob('./*') → 경로명도
# import os  # img_list = os.listdir('./') → 파일명만

data_path = 'C:/Users/kate1/Desktop/eedesign/wgisd-master-data/data'

# npz_list = glob.glob(data_path + '/*.npz')
img_list = []
# txt_list = []
txt_list = glob.glob(data_path + '/*.txt')
for i in txt_list:
    img_list.append(i[:-4] + '.jpg')
    # txt_list.append(i[:-4] + '.txt')

total_num = len(img_list)
# train_label = []  ##
# test_label = []  ##
test_cnt = 0
for i in range(total_num):
    img = Image.open(img_list[i])
    width, height = img.size  # 2048(W) x 1365(H)
    # masked_data = np.load(npz_list[i])
    box_data = np.loadtxt(txt_list[i], dtype='float')
    box_data = box_data[:, 1:]
    # cut image with masked_data
    # masked_data = masked_data['arr_0']  # (H, W, # of class)
    cnt = 0
    if i < 50:
        upper = 1
    else:
        upper = 2
    while cnt < upper:
        center_x = random.randint(256, width-255)
        center_y = random.randint(256, height-255)
        l_b = center_x - 256
        r_b = center_x + 256
        u_b = center_y - 256
        d_b = center_y + 256
        # if masked_data[u_b:d_b+1, l_b:r_b+1, :].sum(axis=2).sum(axis=1).sum(axis=0) > 0:  ##
        if True:  ##
            # cut image
            cropped_img = img.crop((l_b, u_b, r_b, d_b))
            # cropped_img.show()
            # cut label
            # cropped_msk = masked_data[center_y-256:center_y+256, center_x-256:center_x+256, :]
            cropped_box = []
            for box in box_data:
                box_ = box
                # x1 = (box_data[i, 0] - box_data[i, 2] / 2) * width
                # x2 = (box_data[i, 0] + box_data[i, 2] / 2) * width
                # y1 = (box_data[i, 1] - box_data[i, 3] / 2) * height
                # y2 = (box_data[i, 1] + box_data[i, 3] / 2) * height
                box[0], box[2] = 2048*box[0], 2048*box[2]  # width
                box[1], box[3] = 1365*box[1], 1365*box[3]  # height
                if (l_b-box[2]/2 < box[0] < r_b+box[2]/2) and (u_b-box[3]/2 < box[1] < d_b+box[3]/2):
                    x, y, w, h = box[0] - box[2]/2, box[1] - box[3]/2, box[2], box[3]
                    if box[0] < l_b + box[2]/2:
                        w = (box[0] + box[2]/2) - l_b
                        if w > 512:
                            w = 512
                        x = l_b
                    elif box[0] > r_b - box[2]/2:
                        w = r_b - (box[0] - box[2]/2)
                        if w > 512:
                            w = 512
                        x = r_b - w
                    if box[1] < u_b + box[3]/2:
                        h = (box[1] + box[3]/2) - u_b
                        if h > 512:
                            h = 512
                        y = u_b
                    elif box[1] > d_b - box[3]/2:
                        h = d_b - (box[1] - box[3]/2)
                        if h > 512:
                            h = 512
                        y = d_b - h
                    box_[0], box_[1], box_[2], box_[3] = round(x-l_b), round(y-u_b), round(w), round(h)
                    cropped_box.append(box_)
            #####
            # num_msk = cropped_msk.sum(axis=0).sum(axis=0)  ##
            # num_msk = np.count_nonzero(num_msk)  ##
            # if num_msk > 9:  ##
            #    num_msk = 9  ##
            # cropped_msk = cropped_msk.sum(axis=2)
            # cropped_msk[cropped_msk > 0] = 1
            # cropped_msk = cropped_msk.flatten()
            #####
            # save image and label
            save_path = 'C:/Users/kate1/Desktop/eedesign/grape_ss/data'
            if i < 50:  # test
                for box in box_data:
                    if box[2]*box[3] > 10000:
                        cropped_img = img.crop((box[0]-box[2]/2,box[1]-box[3]/2,box[0]+box[2]/2,box[1]+box[3]/2))
                        cropped_img.save(save_path + '/test/bimage/' + str(test_cnt).zfill(4) + '.jpg')
                        test_cnt += 1
                # np.save(save_path + '/test/label/' + str(i).zfill(4) + '.npy', cropped_msk)
                # test_label.append(num_msk)  ##
                #with open(save_path+'/test/label/'+str(i).zfill(4)+'.txt', 'w') as f:
                #    for box in cropped_box:
                #        for xy in box:
                #            f.write(str(xy) + ' ')
                #        f.write('\n')
            else:  # train
                cropped_img.save(save_path + '/train/image/' + str((i-50)*2+cnt).zfill(4) + '.jpg')
                # np.save(save_path + '/train/label/' + str((i-50)*2+cnt).zfill(4) + '.npy', cropped_msk)
                # train_label.append(num_msk)  ##
                with open(save_path+'/train/label/'+str((i-50)*2+cnt).zfill(4)+'.txt', 'w') as f:
                    for box in cropped_box:
                        for xy in box:
                            f.write(str(xy) + ' ')
                        f.write('\n')
            # next stage
            cnt += 1

    print(i, '/', total_num-1)
# np.save('C:/Users/kate1/Desktop/eedesign/cropped/train_label.npy', train_label)  ##
# np.save('C:/Users/kate1/Desktop/eedesign/cropped/test_label.npy', test_label)  ##

# 추가할 요소: 중복 box 제거, box가 하나도 detect 되지 않은 경우 보완
'''

img_path = 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/test/image'
lab_path = 'C:/Users/kate1/Desktop/eedesign/grape_ss/data/test/label'

img_list = glob.glob(lab_path + '/*.jpg')
lab_list = glob.glob(lab_path + '/*.txt')

total_num = len(img_list)
# train_label = []  ##
# test_label = []  ##
for i in range(total_num):
    img = Image.open(img_list[i])
    width, height = img.size  # 2048(W) x 1365(H)
    # masked_data = np.load(npz_list[i])
    box_data = np.loadtxt(lab_list[i], dtype='float')
    box_data = box_data[:, 1:]
'''
