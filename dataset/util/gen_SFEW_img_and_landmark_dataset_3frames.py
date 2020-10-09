
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
from mtcnn.mtcnn import MTCNN
os.chdir(sys.path[0])

# 从原始数据集读取特征和标签 取单帧
def read_features_labels(root_path):
    emo_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5}

    if not os.path.isdir(root_path):
        raise FileNotFoundError()

    features_img = []
    features_lm_geo = []
    features_lm_hog = []
    labels = []

    cnt = 0
    with os.scandir(root_path) as AFEW:
        for emotion in AFEW:
            if emotion.name in emo_dict:
                label = emo_dict[emotion.name]
                with os.scandir(emotion.path) as fold:
                    for sample in fold:
                        with os.scandir(sample) as f:
                            f_list = list(f)
                            frame_num = len(f_list)

                            
                            img0 = f_list[frame_num // 2 - 1]
                            img1 = f_list[frame_num // 2]
                            img2 = f_list[frame_num // 2 + 1]

                            cnt += 1
                            print(cnt, img0.path, label)

                            img_array2, points2_g, points2_h = face_align_and_landmark(img2.path)  # 检测landmark
                            img_array1, points1_g, points1_h = face_align_and_landmark(img1.path)  # 检测landmark
                            img_array0, points0_g, points0_h = face_align_and_landmark(img0.path)  # 检测landmark

                            # return 
                            # features_img.append(np.stack([img_array0, img_array1, img_array2], axis=-1))
                            # features_lm_geo.append(np.stack([points0_g, points1_g, points2_g], axis=1))
                            # features_lm_hog.append(np.stack([points0_h, points1_h, points2_h], axis=1))
                                # print(features.shape, label)
                                # return
                            labels.append(label)

                            if cnt == 5:
                                return 

    return features_img, features_lm_geo, features_lm_hog, labels
                        

def face_align_and_landmark(img_path):
    image_array = cv2.imread(img_path)
    
    casc_path = 'E:\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    
    faceCascade = cv2.CascadeClassifier(casc_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        image_array,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    print(faces)

    x, y, w, h = faces[0]
    left, top, right, bottom = x, y, x + w, y + h
    plt_image = Image.fromarray(image_array)
    plt_image = plt_image.crop((left, top, right, bottom))
    plt.imshow(plt_image)
    plt.show() 
    return 0, 0, 0
    # detector = MTCNN()
    # res_detect = detector.detect_faces(image_array)
    # print(img_path, res_detect)
    # x, y, width, height = res_detect[0]['box']
    # left, top, right, bottom = x, y, x + width, y + height
    # plt_image = Image.fromarray(image_array)
    # plt_image = plt_image.crop(left, top, right, bottom)
    # plt.imshow(plt_image)
    # plt.show() 
    # return 0, 0, 0
    # image_array = face_recognition.load_image_file(img_path)
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
    save_path = '../../data/SFEW_6_classes_img_and_55_landmark_3frames_A+G_train.pickle'
    # label_abs_path = 'G:/dataset/CKplus10G/label0to6.txt'
    root_path = 'G:/dataset/AFEW_img/Train_AFEW'
    label_path = 'G:/dataset/OuluCasIA/OriginalImg/VL/fusion_label6strong.txt'
    test_path = '../../data/oulu_6_classes_img_and_55_landmark_test.pickle'

    # test
    # face_align_and_landmark('G:/dataset/OuluCasIA/OriginalImg/VL/Strong\P013\Disgust\\005.jpeg')
    # exit()

    features_img, features_lm_geo, features_lm_hog, labels = read_features_labels(root_path)
    len_data = len(labels)
    print(len_data)
    # 打乱并分 10 fold
    
    # 存入 pickle 文件
    with open(save_path, 'wb') as pfile:
        pickle.dump(
            {
                'img': features_img,
                'landmark_geo': features_lm_geo,
                'landmark_hog': features_lm_hog,
                'labels': labels
            },
            pfile, pickle.HIGHEST_PROTOCOL
        )
    # # 测试
    # with open(test_path, 'wb') as pfile:
    #     pickle.dump(
    #         data_fold,
    #         pfile, pickle.HIGHEST_PROTOCOL
    #     )