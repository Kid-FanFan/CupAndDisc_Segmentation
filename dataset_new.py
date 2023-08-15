#from cv2.cv2 import transform
from torch.utils.data import Dataset
# from PIL import Image
import torch
# import config
# import torchvision.transforms as transforms
import numpy as np
# import os.path as ops
import json
#import cv2
import os
import copy
import imageio as io
from torchvision.transforms import transforms

from utils import files_with_ext,keep_img_size

root_path = 'data/train_data'
root_path_test = '/media/lab9102/FA0DAF2C6CB5A0CE/tusimple/test_set/'
#list1 = [1, 5, 10, 15, 20] #index-1
# list1 = [2, 5, 9, 14, 20]#index-2
# list1 = [4, 8, 12, 16, 20]#index-3
# list1 = [6, 8, 11, 15, 20]#index-4
# list1 = [8, 11, 14, 17, 20]#index-5
# list1 = [10, 11, 13, 16, 20]#index-6
# list1 = [12, 14, 16, 18, 20]#index-7

class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms,labelP):
        self.img_list = files_with_ext(file_path+"/train_data",".png")
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
        self.path = file_path
        self.labelPath = labelP
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        pic_name = self.img_list[idx]
        path_label = os.path.join(self.path,self.labelPath,pic_name)
        path_img = os.path.join(self.path,"train_data",pic_name)
        label_img = keep_img_size(path_label)
        train_img = keep_img_size(path_img)
        return self.transforms(train_img), self.transforms(label_img)
        # return torch.from_numpy(img), torch.from_numpy(mask)


if __name__ == '__main__':
    op_tranforms = transforms.Compose([transforms.ToTensor(), ])
    data = RoadSequenceDatasetList("data",op_tranforms,"GS1_label/OD_Cup_label")
    print(data[0][0].shape)