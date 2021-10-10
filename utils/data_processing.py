import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, label_filename, ind_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        ind_filepath = os.path.join(data_path, ind_filename)
        fp = open(ind_filepath, 'r')
        self.ind_list = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        ind = int(self.ind_list[index]) - 1
        img = Image.open(os.path.join(self.img_path, self.img_filename[ind]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[ind])
        return img, label, index

    def __len__(self):
        return len(self.ind_list)
