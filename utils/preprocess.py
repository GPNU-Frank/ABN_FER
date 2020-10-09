import os
import pickle 
import cv2
import numpy as np


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = []
    for i in range(num_classes):
        if i == labels_dense:
            labels_one_hot.append(1)
        else:
            labels_one_hot.append(0)
    return labels_one_hot


def pickle_2_img_single(data_file):
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x1, total_y = [], []
    for i in range(len(data)):
        x1 = []
        yl = []
        print(len(data[i]['img']))
        for j in range(len(data[i]['labels'])):
             
            img = data[i]['img'][j]
            # img = ImageBlocks.getImageBlocks(img, 16, 16)
            
            label = int(data[i]['labels'][j])
             
            if label == 7:
                label = 2
            # print(label)
            # label = dense_to_one_hot(label, 6)
             
            #print(label)
            x1.append(img)
            yl.append(label)

        total_x1.append(x1)
        total_y.append(yl)    
           
    # exit(1)
    return total_x1, total_y

def pickle_2_img_and_landmark(data_file):
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x1, total_lm, total_y = [], [], []
    for i in range(len(data)):
        x1 = []
        lm1 = []
        yl = []
        print(len(data[i]['img']))
        for j in range(len(data[i]['labels'])):
             
            img = data[i]['img'][j]
            # img = ImageBlocks.getImageBlocks(img, 16, 16)
            
            landmark = data[i]['landmark'][j]
            label = int(data[i]['labels'][j])
             
            if label == 7:
                label = 2
            # print(label)
            # label = dense_to_one_hot(label, 6)
             
            #print(label)
            x1.append(img)
            lm1.append(landmark)
            yl.append(label)

        total_x1.append(x1)
        total_lm.append(lm1)
        total_y.append(yl)    
           
    # exit(1)
    return total_x1, total_lm, total_y


def pickle_2_img_and_landmark_ag(data_file):
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x1, total_lm_g, total_lm_h, total_y = [], [], [], []
    for i in range(len(data)):
        x1 = []
        lmg1 = []
        lmh1 = []
        yl = []
        print(len(data[i]['img']))
        for j in range(len(data[i]['labels'])):
             
            img = data[i]['img'][j]
            # img = ImageBlocks.getImageBlocks(img, 16, 16)
            
            landmark_g = data[i]['landmark_geo'][j]
            landmark_h = data[i]['landmark_hog'][j]
            label = int(data[i]['labels'][j])
             
            if label == 7:
                label = 2
            # print(label)
            # label = dense_to_one_hot(label, 6)
             
            #print(label)
            x1.append(img)
            lmg1.append(landmark_g)
            lmh1.append(landmark_h)
            yl.append(label)

        total_x1.append(x1)
        total_lm_g.append(lmg1)
        total_lm_h.append(lmh1)
        total_y.append(yl)    
           
    # exit(1)
    return total_x1, total_lm_g, total_lm_h, total_y


def pickle_2_img_and_landmark_and_crop(data_file):
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x1, total_lm, total_leye, total_reye, total_nose, total_lip, total_y = [], [], [], [], [], [], []
    for i in range(len(data)):
        x1 = []
        lm1 = []
        le1 = []
        re1 = []
        ns1 = []
        lp1 = []
        yl = []
        print(len(data[i]['img']))
        for j in range(len(data[i]['labels'])):
             
            img = data[i]['img'][j]
            # img = ImageBlocks.getImageBlocks(img, 16, 16)
            
            landmark = data[i]['landmark'][j]

            leye = data[i]['leye'][j]

            reye = data[i]['reye'][j]

            nose = data[i]['nose'][j]

            lip = data[i]['lip'][j]

            label = int(data[i]['labels'][j])
             
            if label == 7:
                label = 2
            # print(label)
            # label = dense_to_one_hot(label, 6)
             
            #print(label)
            x1.append(img)
            lm1.append(landmark)
            le1.append(leye)
            re1.append(reye)
            ns1.append(nose)
            lp1.append(lip)
            yl.append(label)

        total_x1.append(x1)
        total_lm.append(lm1)
        total_leye.append(le1)
        total_reye.append(re1)
        total_nose.append(ns1)
        total_lip.append(lp1)
        total_y.append(yl)    
           
    # exit(1)
    return total_x1, total_lm, total_leye, total_reye, total_nose, total_lip, total_y