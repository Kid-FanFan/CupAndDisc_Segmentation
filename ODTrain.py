import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from dataset_new import *
from UModel import *
from torchvision.utils import save_image
#import tensorflow as tf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path= 'ODparams/OD.pth'
data_path = '/home/lenovo/Glaucoma_Segmentation_Unet/data'
save_path = '/home/lenovo/Glaucoma_Segmentation_Unet/ODResult'
if __name__ == '__main__':
    op_tranforms = transforms.Compose([transforms.ToTensor(), ])
    data_loader = DataLoader(RoadSequenceDatasetList(data_path,op_tranforms,"OD_label"),batch_size=3,shuffle=True)
    net = Unet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("success loadd weight")
    else:
        print("False load weight")

    opt = torch.optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    while True:
        for i,(data_img,label_image) in enumerate(data_loader):
            data_img,label_image = data_img.to(device),label_image.to(device)

            out_image = net(data_img)
            train_loss = loss_fun(out_image,label_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%50==0:
                torch.save(net.state_dict(),weight_path)

            _image = data_img[0]
            _label = label_image[0]
            _out_image = out_image[0]

            save_img=torch.stack([_image,_label,_out_image],dim=0)
            save_image(save_img,f'{save_path}/{i}.png')

        epoch+=1