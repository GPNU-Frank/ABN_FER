import cv2
import os


def batch_covert(root_path):


    emo_dict = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sadness': 4, 'Surprise': 5}
    if not os.path.isdir(root_path):
        raise FileNotFoundError
    
    cnt = 0
    with os.scandir(root_path) as AFEW:
        for AFEW_folder in AFEW:   # train/val/test
            if os.path.isdir(AFEW_folder.path):
                with os.scandir(AFEW_folder.path) as dataset:
                    for emotion in dataset: # angry/disgust/..
                        if os.path.isdir(emotion.path):
                            with os.scandir(emotion.path) as fold:
                                for img_path in fold:
                                    # print(img_path.path)
                                    gen_path = img_path.path.replace('AFEW', 'AFEW_img', 1)
                                    # fold_path = '\\'.join(gen_path.split('\\')[:-1])
                                    fold_path = gen_path[:-4]
                                    print(fold_path)

                                    if not os.path.exists(fold_path):
                                        os.makedirs(fold_path)
                                    os.system('C:\\Users\\Frank\\Downloads\\ffmpeg-4.3.1-2020-10-01-essentials_build\\bin\\ffmpeg.exe -i ' + img_path.path + ' -qscale:v 1 -f image2 ' + fold_path + '\\%06d.jpg')

                                    cnt += 1

                                    # if cnt == 5:
                                    #     return


#  利用opencv的VideoCapture从TownCentreXVID.avi中抽取用于训练的图像帧，存储到JPEGImages和JPEGImages_test
def video2im(video_path, root_path='G:\\dataset', train_path='JPEGImages', gen_path='JPEGImages_test', factor = 2):
    frame = 0
    cap = cv2.VideoCapture(video_path)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Total Frame Count:', length)
    print(fps)
    gen_path = video_path.replace('AFEW', 'AFEW_img', 1)
    
    fold_path = '\\'.join(gen_path.split('\\')[:-1])

    print(fold_path)

    # if not os.path.exists(fold_path):
    #     os.makedirs(fold_path)
    while True:

        check, img = cap.read()
        print(frame, check, img.shape)
        frame += 1
        # if check:
        #     # img = cv2.resize(img, (1920 // factor, 1080 // factor))
        #     cv2.imwrite(os.path.join(gen_path, str(frame) + ".jpg"), img)
        #     frame += 1
        #     print('Processed: ',frame)
        # else:
        #     break
        if frame == 10:
            break    
    cap.release()


if __name__ == '__main__':
    
    root_path = 'G:/dataset/AFEW'

    batch_covert(root_path)