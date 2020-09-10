
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
def read_features_labels(root_path):
    emo_dict = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sadness': 4, 'Surprise': 5}
    optical_condition = {'Dark': 0, 'Strong': 1, 'Weak': 2}
    tripet_data = []  # (img, landmark, label) 用于分fold

    for emotion, label in emo_dict.items():
        emotion_folder_path = root_path + emotion

        person_data = [[[] for j in range(80) ] for i in range(3)]  # 放每个人的所有图片
        with os.scandir(emotion_folder_path) as e_f:
            for img in e_f:  # dark person_20 为空
                img_name = img.name[:-5]  # 去掉 jpeg
                name_split = img_name.split('_')
                person_num = int(name_split[1][1:]) - 1  # Dark_P001_Anger_008
                optical_num = optical_condition[name_split[0]]
                img_path = img.path
                # if not img_path:
                #     raise ValueError()
                print(img_name, img_path)
                person_data[optical_num][person_num] = img_path

            for i in range(3):
                for j in range(80):
                    if person_data[i][j]:
                        img_array, landmark = face_align_and_landmark(person_data[i][j])
                        tripet_data.append((img_array, landmark, label))         
    return tripet_data
                        

def face_align_and_landmark(img_path):
    if not img_path:
        return
    image_array = cv2.imread(img_path)
    face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
    face_landmarks_dict = face_landmarks_list[0]
    cropped_face, left, top = corp_face(image_array, face_landmarks_dict)
    transferred_landmarks = transfer_landmark(face_landmarks_dict, left, top)
    cropped_face, transferred_landmarks = resize_img_and_landmark(cropped_face, transferred_landmarks)
    
    # 除去下颚的特征点,一共55个特征点
    list_landmarks = []
    for key in transferred_landmarks.keys():
        if key != 'chin':
            list_landmarks.extend(transferred_landmarks[key])
    # print(len(list_landmarks))
    # plt.imshow(cropped_face)
    # plt.show()

    # visualize_landmark(cropped_face, list_landmarks)
    return cropped_face[:, :, 0], np.stack(list_landmarks, axis=1)


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
    save_path = '../../data/oulu_6_classes_img_and_55_landmark.pickle'
    # label_abs_path = 'G:/dataset/CKplus10G/label0to6.txt'
    root_path = 'G:/dataset/Oulu_CasIA_Img_NL_ClassifiedFiltered_byTY_YYS/'
    test_path = '../../data/oulu_6_classes_img_and_55_landmark_test.pickle'

    tripet_data = read_features_labels(root_path)
    # total 1440 (3 * 480) 可能缺少几个样本
    # updata: total 1186
    len_data = len(tripet_data) 
    print(len_data)
    # 打乱并分 10 fold
    random.shuffle(tripet_data)
    data_fold = []
    fold, fold_num = 10, 119
    # start = 0
    for i in range(0, len_data, fold_num):
        data_fold.append(tripet_data[i: min(i + fold_num, len_data)])  # (10, 119, 3)
    data_fold_transpose = [[[] for j in range(3) ] for i in range(10)]  # 放每个人的所有图片
    for fold in range(10):
        for in_fold in range(len(data_fold[fold])):
            for tri_index in range(3):
                data_fold_transpose[fold][tri_index].append(data_fold[fold][in_fold][tri_index])
    # print(len(tripet_data))  
    # print(tripet_data[0][0].shape, tripet_data[0][1].shape, tripet_data[0][2])
    # exit(0) 
    

    # 存入 pickle 文件
    with open(save_path, 'wb') as pfile:
        pickle.dump(
            [{
                'img': data_fold_transpose[i][0],
                'landmark': data_fold_transpose[i][1],
                'labels': data_fold_transpose[i][2]
            } for i in range(10)],
            pfile, pickle.HIGHEST_PROTOCOL
        )
    # 测试
    with open(test_path, 'wb') as pfile:
        pickle.dump(
            data_fold,
            pfile, pickle.HIGHEST_PROTOCOL
        )