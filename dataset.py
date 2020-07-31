# -*- coding: utf-8 -*- 
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #self.task = task
        self.opts = opts
        

        lst_dir = os.listdir(data_dir) # female / male
        
        self.male_dir = os.path.join(lst_dir, "male")
        self.female_dir = os.path.join(lst_dir, "female")
        
        male_data = [f for f in os.listdir(male_dir) if f.endswith('jpg') | f.endswith('png')]
        female_data = [f for f in os.listdir(female_dir) if f.endswith('jpg') | f.endswith('png')]

        self.lst_data = male_data + female_data


        self.male_data = male_data
        self.female_data = female_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        
        if os.path.isdir(female_dir):
            label = 0
        elif os.path.isdir(male_dir):
            label = 1
        

        img = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape


        # channel axis 추가해주기
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        data = {'label': label, 'input' : img }

        if self.transform:
            data=self.transform(data)
        

        return data




