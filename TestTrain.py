# -*- coding: utf-8 -*-
# @time : 2022/4/2 10:04
# @author : Precision
# @file : GlaucomaTrain.py
# @project : Glaucoma_Segmentation_Unet
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from dataset_new import *
#from UModel import *
#from OUModel import *
#from DUModel import *
from MaxMouble import *
from torchvision.utils import save_image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 填写你要使用的GPU号
#import tensorflow as tf
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#weight_path='params/unet.pth'
#weight_path= 'params1/unet1.pth'  #DU
#weight_path='params/unet.pth' #OU
weight_path='params/unet5.pth' #OU
data_path = '/home/lenovo/Glaucoma_Segmentation_Unet/data'
save_path = '/home/lenovo/Glaucoma_Segmentation_Unet/resultTest'
if __name__ == '__main__':
    op_tranforms = transforms.Compose([transforms.ToTensor(), ])
    data_loader = DataLoader(RoadSequenceDatasetList(data_path,op_tranforms,"GS1_label/OD_Cup_label"),batch_size=5,shuffle=True)
    net = Unet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("success loadd weight")
    else:
        print("False load weight")

    opt = torch.optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()
    epoch = 1
    BestLoss = 0.008
    while epoch < 2:
        LossNum = 0
        for i,(data_img,label_image) in enumerate(data_loader):
            data_img,label_image = data_img.to(device),label_image.to(device)
            out_image = net(data_img)
            train_loss = loss_fun(out_image,label_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%3==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # if i%8==0:
            #     print("save loss")
            #     torch.save(net.state_dict(),weight_path)
            LossNum = LossNum+train_loss.item()
            _image = data_img[0]
            _label = label_image[0]
            _out_image = out_image[0]

            save_img=torch.stack([_image,_label,_out_image],dim=0)
            save_image(save_img,f'{save_path}/{i}.png')
        LossNum = LossNum/10
        if LossNum < BestLoss :
            BestLoss = LossNum
            print(BestLoss,"       ==>Save Loss")
            torch.save(net.state_dict(),weight_path)

        epoch+=1