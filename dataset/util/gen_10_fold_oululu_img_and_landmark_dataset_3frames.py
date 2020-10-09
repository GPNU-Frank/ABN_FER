
import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import dlib
from sklearn.model_selection import StratifiedKFold
import face_recognition
from collections import defaultdict
import random
import math
from skimage.feature import hog
os.chdir(sys.path[0])

# 从原始数据集读取特征和标签 取单帧
def read_features_labels(label_abs_path, root_path):
    emo_dict = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sadness': 4, 'Surprise': 5}
    optical_condition = {'Dark': 0, 'Strong': 1, 'Weak': 2}
    tripet_data = []  # (img, landmark, label) 用于分fold

    if not os.path.isfile(label_abs_path):
        raise FileNotFoundError()

    features_img = [[] for _ in range(480)]
    features_lm_geo = [[] for _ in range(480)]
    features_lm_hog = [[] for _ in range(480)]
    labels = [[] for _ in range(480)]
    with open(label_abs_path, 'r') as f:
        lines = f.readlines()

        cnt = 0

        for idx, line in enumerate(lines):
            img_path, label = line.split()

            _, person, emotion, number = img_path.split('\\')

            person = int(person[1:])
            label = int(label)
            img_abs_path = root_path + '/' + img_path

            cnt += 1

            print(cnt, img_abs_path, label)

            img_array, point_g, point_h = face_align_and_landmark(img_abs_path)

            features_img[idx // 3].append(img_array)
            features_lm_geo[idx // 3].append(point_g)
            features_lm_hog[idx // 3].append(point_h)
            labels[idx // 3] = label
            
            # if cnt == 36:
            #     features_img = features_img[: (cnt + 1) // 3]
            #     features_lm_geo = features_lm_geo[: (cnt + 1) // 3]
            #     features_lm_hog = features_lm_hog[: (cnt + 1) // 3]
            #     labels = labels[:(cnt + 1) // 3]
            #     break
                
        features_img = np.array(features_img)
        features_lm_geo = np.array(features_lm_geo)
        features_lm_hog = np.array(features_lm_hog)
        labels = np.array(labels)
        features_img = features_img.transpose((0, 2, 3, 1))
        features_lm_geo = features_lm_geo.transpose((0, 2, 1, 3))
        features_lm_hog = features_lm_hog.transpose((0, 2, 1, 3))
        # labels = labels.reshape((-1, 3))

    return list(zip(features_img, features_lm_geo, features_lm_hog, labels))
                        

def face_align_and_landmark(img_path):
    image_array = cv2.imread(img_path)
    face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
    face_landmarks_dict = face_landmarks_list[0]
    cropped_face, left, top = corp_face(image_array, face_landmarks_dict)
    transferred_landmarks = transfer_landmark(face_landmarks_dict, left, top)
    cropped_face, transferred_landmarks = resize_img_and_landmark(cropped_face, transferred_landmarks)
    
    # 灰度化
    cropped_face = np.array(Image.fromarray(cropped_face).convert('L'))

    # print(transferred_landmarks)
    # return
    # 除去下颚的特征点,一共55个特征点
    # list_landmarks = []
    # for key in transferred_landmarks.keys():
    #     if key != 'chin':
    #         list_landmarks.extend(transferred_landmarks[key])
    list_landmarks = []
    hog_landmarks = []

    # hog 特征
    normalised_blocks, hog_image = hog(cropped_face, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, feature_vector=False)
    
    bin_size = 6
    for key in transferred_landmarks.keys():
        if key != 'chin':
            for x, y in transferred_landmarks[key]:
                # list_landmarks.extend((x,y))

                # x, y 和 patch 分开
                patch_group = []
                block_y = y // bin_size
                block_x = x // bin_size
                list_landmarks.append([x, y])
                hog_landmarks.append(normalised_blocks[block_y][block_x].flatten())
    # print(len(list_landmarks))
    # plt.imshow(cropped_face)
    # plt.show()

    # visualize_landmark(cropped_face, list_landmarks)
    return cropped_face, np.stack(list_landmarks, axis=1), np.stack(hog_landmarks, axis=1)


def resize_img_and_landmark(image_array, landmarks):
    img_crop = Image.fromarray(image_array)
    ori_size = img_crop.size
    img_crop = img_crop.resize((224, 224))
    ratio = (224 / ori_size[0], 224 / ori_size[1])
    transferred_landmarks = defaultdict(list)

    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (int(landmark[0] * ratio[0]), int(landmark[1] * ratio[1]))
            transferred_landmarks[facial_feature].append(transferred_landmark)
    img_crop = np.array(img_crop)
    return img_crop, transferred_landmarks


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for points in landmarks:
        draw.point(points)
    plt.imshow(origin_img)
    plt.show()

def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

if __name__ == '__main__':
    # 配置参数
    # file_name = '../../data/cohn-kanade-images/'
    save_path = '../../data/oulu_6_classes_img_and_55_landmark_3frames_A+G.pickle'
    # label_abs_path = 'G:/dataset/CKplus10G/label0to6.txt'
    root_path = 'G:/dataset/OuluCasIA/OriginalImg/VL'
    label_path = 'G:/dataset/OuluCasIA/OriginalImg/VL/fusion_label6strong.txt'
    test_path = '../../data/oulu_6_classes_img_and_55_landmark_test.pickle'

    # test
    # face_align_and_landmark('G:/dataset/OuluCasIA/OriginalImg/VL/Strong\P013\Disgust\\005.jpeg')
    # exit()

    tripet_data = read_features_labels(label_path, root_path)
    len_data = len(tripet_data)
    print(len_data)
    # 打乱并分 10 fold
    random.shuffle(tripet_data)
    data_fold = []
    fold, fold_num = 10, 48
    # start = 0
    for i in range(0, len_data, fold_num):
        data_fold.append(tripet_data[i: min(i + fold_num, len_data)])  # (10, 119, 3)
    data_fold_transpose = [[[] for j in range(4) ] for i in range(10)]  # 放每个人的所有图片
    for fold in range(10):
        for in_fold in range(len(data_fold[fold])):
            for tri_index in range(4):
                data_fold_transpose[fold][tri_index].append(data_fold[fold][in_fold][tri_index])
    

    # 存入 pickle 文件
    with open(save_path, 'wb') as pfile:
        pickle.dump(
            [{
                'img': data_fold_transpose[i][0],
                'landmark_geo': data_fold_transpose[i][1],
                'landmark_hog': data_fold_transpose[i][2],
                'labels': data_fold_transpose[i][3]
            } for i in range(10)],
            pfile, pickle.HIGHEST_PROTOCOL
        )
    # # 测试
    # with open(test_path, 'wb') as pfile:
    #     pickle.dump(
    #         data_fold,
    #         pfile, pickle.HIGHEST_PROTOCOL
    #     )