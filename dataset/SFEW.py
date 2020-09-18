from PIL import Image
from torch.utils.data import Dataset
import os


class SFEW(Dataset):
    def __init__(self, root, transform=None, train=True): 
    
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



if __name__ == '__main__':
    train_path = '../data/SFEW/Train/Train_Aligned_Faces/'
    test_path = '../data/SFEW/Test/Test_Aligned_Faces/'

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    ])

    trian_set = SFEW(root=train_path, transform=transform, train=True)
    print(len(trian_set))
    train