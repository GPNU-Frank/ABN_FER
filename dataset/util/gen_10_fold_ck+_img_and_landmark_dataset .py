
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
os.chdir(sys.path[0])

# 从原始数据集读取特征和标签 取单帧
def read_features_labels(label_abs_path, root_path):
    cnt = 0
    # min_frame_num = 100
    # min_path = ''
    if not os.path.isfile(label_abs_path):
        raise FileNotFoundError()
    features_img = [[] for i in range(10)]
    features_lm = [[] for i in range(10)]
    labels = [[] for i in range(10)]

    with open(label_abs_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            img_path, label = line.split()
            fold_index = int(img_path.split('/')[0][1:]) - 1
            print(img_path, fold_index, label)
            # return
            label = int(label) - 1
            if label == -1:  # 6 classes 分类
                continue
            img, landmark = face_align_and_landmark(root_path + img_path)
            features_img[fold_index].append(img)
            features_lm[fold_index].append(landmark)
            labels[fold_index].append(label)
            # print(img.shape, landmark.shape, label)
            # return
            
    return features_img, features_lm, labels
                        

def face_align_and_landmark(img_path, img_size=(128, 128)):
    rows, cols = img_size

    image_array = cv2.imread(img_path)
    face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
    face_landmarks_dict = face_landmarks_list[0]
    # print(face_landmarks_dict)
    # return
    cropped_face, left, top = corp_face(image_array, face_landmarks_dict)
    transferred_landmarks = transfer_landmark(face_landmarks_dict, left, top)
    cropped_face, transferred_landmarks = resize_img_and_landmark(cropped_face, transferred_landmarks)
    
    # 灰度化
    cropped_face = np.array(Image.fromarray(cropped_face).convert('L'))

    # 除去下颚的特征点,一共55个特征点
    list_landmarks = []
    for key in transferred_landmarks.keys():
        if key != 'chin':
            list_landmarks.extend(transferred_landmarks[key])
    # print(len(list_landmarks))
    # plt.imshow(cropped_face)
    # plt.show()

    # 特征点转下标
    lm_index = []

    for j in range(len(list_landmarks)):
        patch_group = []
        patch_group.append(list_landmarks[j, 1] * cols + list_landmarks[j])  # 中心
        patch_group.append((list_landmarks[j, 1] - 1) * cols + list_landmarks[j, 0] - 1)  # 左上角
        patch_group.append((list_landmarks[j, 1] - 1) * cols + list_landmarks[j, 0])  # 上方
        patch_group.append((list_landmarks[j, 1] - 1) * cols + list_landmarks[j, 0] + 1)  # 右上角
        patch_group.append((list_landmarks[j, 1]) * cols + list_landmarks[j, 0] - 1)  # 左侧
        patch_group.append((list_landmarks[j, 1]) * cols + list_landmarks[j, 0] + 1)  # 右侧
        patch_group.append((list_landmarks[j, 1] + 1) * cols + list_landmarks[j, 0] - 1)  # 左下角
        patch_group.append((list_landmarks[j, 1] + 1) * cols + list_landmarks[j, 0])  # 下方
        patch_group.append((list_landmarks[j, 1] + 1) * cols + list_landmarks[j, 0] + 1)  # 右下角

    # visualize_landmark(cropped_face, list_landmarks)
    return cropped_face, np.stack(list_landmarks, axis=1)


def resize_img_and_landmark(image_array, landmarks):
    img_crop = Image.fromarray(image_array)
    ori_size = img_crop.size
    img_crop = img_crop.resize((128, 128))
    ratio = (128 / ori_size[0], 128 / ori_size[1])
    transferred_landmarks = defaultdict(list)

    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (int(landmark[0] * ratio[0]), int(landmark[1] * ratio[1]))
            transferred_landmarks[facial_feature].append(transferred_landmark)
    img_crop = np.array(img_crop)
    return img_crop, transferred_landmarks

def visualize_landmark_ori(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
        for id, points in enumerate(landmarks[facial_feature]):
            draw.text(points, str(id))
    plt.imshow(origin_img)

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
    file_name = '../../data/cohn-kanade-images/'
    save_path = '../../data/ck+_6_classes_img_and_55_landmark.pickle'
    label_abs_path = 'G:/dataset/CKplus10G/label0to6.txt'
    root_path = 'G:/dataset/CKplus10G/'

    # 读取数据 原标签是从1开始 所以要减 1, 把标签7改为 类别2, 一共593个表情序列 但能用的只有327个
    feature_img, feature_lm, labels = read_features_labels(label_abs_path, root_path)
    print(len(feature_img), len(feature_lm), len(labels))
    print(len(feature_img[0]), len(feature_lm[0]), len(labels[0]))
    print(feature_img[0][0].shape, feature_lm[0][0].shape, labels[0][0])
    # exit(0) 
    

    # 存入 pickle 文件
    with open(save_path, 'wb') as pfile:
        pickle.dump(
            [{
                'img': feature_img[i],
                'landmark': feature_lm[i],
                'labels': labels[i]
            } for i in range(10)],
            pfile, pickle.HIGHEST_PROTOCOL
        )