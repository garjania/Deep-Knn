import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os


class SeqDataset(Dataset):
    def __init__(self, x_path, label_path, sec=0, ind=1):
        super(SeqDataset, self).__init__()

        self.data = np.load(x_path)
        self.data = np.moveaxis(self.data, [1], [0])
        self.data = np.expand_dims(self.data, axis=1)
        length = int(self.data.shape[0] / 10)
        if sec != -1:
            self.data = self.data[sec * length:(sec + ind) * length, :]
        print(self.data.shape)

        self.labels = np.load(label_path)
        if sec != -1:
            self.labels = self.labels[sec * length:(sec + ind) * length]
        print(self.labels.shape)

        self.data_0 = torch.tensor(self.data[np.where(self.labels == 0)[0]], dtype=torch.float32)
        self.data_1 = torch.tensor(self.data[np.where(self.labels == 1)[0]], dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index]), index

    def __len__(self):
        return len(self.data)


class SkinDataset(Dataset):
    def __init__(self, images, labels):
        super(SkinDataset, self).__init__()

        self.data = images
        self.labels = labels
        self.data_0 = torch.tensor(self.data[self.labels == 0], dtype=torch.float32)
        self.data_1 = torch.tensor(self.data[self.labels == 1], dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index]), index

    def __len__(self):
        return len(self.data)


def resize(img):
    if img.shape[1] > 1024:
        new_width = 1024
        new_height = int(img.shape[1] * 1024/img.shape[1])
        img = cv2.resize(img, (new_width, new_height))
    frame = np.zeros((1024, 1024, 3))
    x_pad = int((1024 - img.shape[0])/2)
    y_pad = int((1024 - img.shape[1])/2)
    frame[x_pad:x_pad+img.shape[0], y_pad:y_pad+img.shape[1], :] = img
    return frame


def load_skin_datasets(img_path, label_path, filter=True):
    df = pd.read_csv(label_path)
    df['label'] = 2 * df['melanoma'] + df['seborrheic_keratosis']
    if filter:
        df = df[df['label'] != 2]
    df = df.to_dict('list')
    maxes = [0,0]
    images = []
    for filename in df['image_id'][:131]:
        img = cv2.imread(os.path.join(img_path, filename) + '.jpg')
        if img is not None:
            img = resize(img)
            img = np.expand_dims(np.moveaxis(img, -1, 0), 0)
            images.append(img)

    images = np.concatenate(images)
    labels = np.array(df['label'][:131], dtype=np.int)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=1)

    train_dataset = SkinDataset(X_train, y_train)
    test_dataset = SkinDataset(X_test, y_test)

    return train_dataset, test_dataset
