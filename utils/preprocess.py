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