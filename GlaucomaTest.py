# -*- coding: utf-8 -*-
# @time : 2022/4/4 14:46
# @author : Precision
# @file : GlaucomaTest.py
# @project : Glaucoma_Segmentation_Unet
import os

import torch
from torchvision import transforms
from torchvision.utils import save_image

from utils import keep_img_size,files_with_ext
#from UModel import Unet
#from DUModel import  Unet
#from OUModel import  Unet
from MaxMouble import Unet

op_tranforms = transforms.Compose([transforms.ToTensor(), ])

net = Unet().cuda()

save_path = "TestImg/max_res"
weights = "params/unet5.pth"
data_path = "data/val_data"
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print("success loadd weight")
else:
    print("False load weight")

def ImgTest(filename,file_path):
    #_input = input("please input your path：")
    out_img = keep_img_size(file_path)
    img_data=op_tranforms(out_img).cuda()
    img_data=torch.unsqueeze(img_data,dim=0)
    out_data=net(img_data)
    save_image(out_data,f'{save_path}/{filename}')

data_list = files_with_ext(data_path,".png")
data_size = len(data_list)
for i in data_list:
    print(i)
    ImgTest(i,os.path.join(data_path,i))
