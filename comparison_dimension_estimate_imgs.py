
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import torch
import skdim

import torchvision.transforms as transforms
import numpy as np

from utils.datasets import ImagesDataset, Jittering

#Set the device
GPU = True
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')

#Transforms
transform = transforms.Compose([transforms.ToTensor(),])

#Generate the dataset

n_data = 512
batch_size = 256

train_dataset = ImagesDataset(epoch_size=n_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),eps=6,fix_int=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

#Flatten the dataset

dataset_list = []

for xs,ids in train_dataloader:
        for im in xs:   
            dataset_list.append(im[0].view(im.shape[0],-1))

dataset = torch.vstack(dataset_list)


#Estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):

lpca = skdim.id.lPCA().fit_pw(dataset,
                              n_neighbors = 100,
                              n_jobs = 1)
print("LPCA ID:", np.mean(lpca.dimension_pw_))

#Estimate global intrinsic dimension with MLE

mle = skdim.id.MLE().fit(dataset)
print("MLE ID:", mle.dimension_)

#Estimate global intrinsic using the correlation dimension

corrdim = skdim.id.CorrInt(k1=50, k2=100, DM=False).fit(dataset)
print("Correlation dimension ID:", corrdim.dimension_)