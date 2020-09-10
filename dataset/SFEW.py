from PIL import Image
from torch.utils.data import Dataset
import os


class SFEW(Dataset):
    def __init__(self, root, transform=None): 
    
        class_num = 7
        label_map = {'Angry': 0, 'Disgust': '1', 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
        imgs = []

        if not os.path.isdir(root):
            raise FileNotFoundError('root is not a folder')

        with os.scandir(root) as folds:
            for emotion in folds:
                label = emotion.name
                with os.scandir(emotion.path) as emotion_fold:
                    for img in emotion_fold:
                        imgs.append((img.path, label_map[label]))

        self.root = root
        self.imgs = imgs
        self.class_num = class_num
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.imgs[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)



