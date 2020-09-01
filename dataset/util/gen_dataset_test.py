
import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import dlib
from sklearn.model_selection import StratifiedKFold


# 从原始数据集读取特征和标签
def read_features_labels(file_name):
    cnt = 0
    # min_frame_num = 100
    # min_path = ''
    if not os.path.isdir(file_name):
        raise FileNotFoundError()
    features = []
    labels = []
    with os.scandir(file_name) as folds:
        for person in folds:  # 每一个人的文件夹 如 S135
            with os.scandir(person.path) as person_emotion:
                for fold in person_emotion:  # 每个人里的表情文件夹
                    # print(dir(fold))
                    if fold.is_dir():
                        label = int(fold.name)
                        # if cnt >= 10:
                        #     break
                        with os.scandir(fold) as f:
                            # f_list = list(f)
                            # frame_num = len(f_list)
                            # frame_extract = [0, (frame_num - 1) // 2, frame_num - 1]
                            features_one = []
                            # frame_index = 0
                            # frame_num = 0
                            for img in f:  # 表情文件夹里的每一帧
                                # print(img.path)
                                # if frame_index in frame_extract:

                                # frame_num += 1
                                img_feature = get_raw_picture(img.path)  # 获取人脸像素
                                return
                                features_one.append(img_feature)
                                # frame_index += 1
                            features_one = np.stack(features_one, axis=1)
                                # print(features.shape, label)
                                # return
                            features.append(features_one)
                            labels.append(label)
                        # if frame_num < min_frame_num:
                        #     min_path = fold.path
                        # min_frame_num = min(min_frame_num, frame_num)
                        cnt += 1
                        # if cnt > 20:
                        #     return np.stack(features), np.stack(labels)
                        print(cnt)
    return np.stack(features), np.stack(labels)
                        


def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# 
def get_raw_picture(img_path):
    img = Image.open(img_path)
    img.
    img = np.array(img)
    print(img.shape)

def precess_image(img_path):
    """
    输入图片路径，裁剪人脸位置并标记68特征点
    """
    img = cv2.imread(img_path)
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except Exception:
        print(img_path)
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        points = []
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            points.append(np.array(pos))

            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 2, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx + 1), pos, font, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
        
        # 裁剪图片
        # (x, y, w, h) = rect_to_bbox(rects[i])
        # img = img[y: y + h, x: x + w]
    # landmark 的边
    # neighbor_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    # (10, 11), (11,12), (12, 13), (13, 14), (14, 15), (15, 16),  # 轮廓
    # (17, 18), (18, 19), (19, 20), (20, 21), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 37),  # 左眼
    # ]

        points = np.stack(points, axis=1)

    # return points
        print(points.shape)

    # 展示
    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
        


if __name__ == '__main__':
    # 配置参数
    file_name = '../../data/cohn-kanade-images/'
    # landmarks_mat_path = '../../resource/shape_predictor_68_face_landmarks.dat'
    save_path = '../../data/ck+_224_224.pickle'
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(landmarks_mat_path)

    # 读取数据
    features, labels = read_features_labels(file_name)
    print(features.shape, labels.shape)


    # 存入 pickle 文件
    with open(save_path, 'wb') as pfile:
        pickle.dump(
            {
                'features': features,
                'labels': labels,
            },
            pfile, pickle.HIGHEST_PROTOCOL
        )