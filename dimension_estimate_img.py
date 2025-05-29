import torch

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from utils.datasets import ImagesDataset, Jittering

#Set the device
GPU = True
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')



def compute_metric(fun,xs):
    #Compute the Jacobian and its transpose
    idxa = [i for i in range(len(xs))]
    idxb = [i for i in range(len(xs))]
    jacobian = torch.autograd.functional.jacobian(fun,xs,vectorize=True) 
    jacobian = torch.cat([jacobian[x, :, y, :].unsqueeze(0) for x, y in zip(idxa, idxb)]).to(device)    
    shape = jacobian.shape
    jacobian = jacobian.reshape(shape[0],shape[1],shape[3]*shape[4])
    shape = jacobian.shape
    jacobian_t = jacobian.swapaxes(-1,-2)

    #Compute the pullback metric
    g = torch.einsum('bij,bjk->bik',jacobian,jacobian_t)

    return g


#Transforms
transform = transforms.Compose([transforms.ToTensor(),])

#Generate the dataset

n_data = 5000
n_test_data = 5000
batch_size = 256

train_dataset = ImagesDataset(epoch_size=n_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),eps=6,fix_int=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

test_dataset = ImagesDataset(epoch_size=n_test_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),test=True,eps=6,fix_int=True)    
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)


# Load the model
model = torch.load("img_0_model.pth")
model.eval()


#Extract samples
imgs = []

nsamples = 500
for idx in range(nsamples):
    tmp = train_dataset.__getitem__(idx)[0][0]
    imgs.append(tmp[None,:,:])

imgs = torch.stack(imgs, dim=0).to(device)
  

#Compute the eigenvalues
elist = []
for i in range(500):
    g = compute_metric(model.reduce,imgs[i][None,:,:,:])
    eigenvalues, _ = torch.linalg.eigh(g)
    elist.append(eigenvalues)


#Compute mean of the eigenvalues vector
es = torch.zeros_like(elist[0])
for ev in elist:
    es += ev
es = es / nsamples

#Sort the average eigenvalues and compute their log10
es = es[0].detach().cpu().numpy()
nb = np.arange(0,len(es),1)
es = np.sort(es)[::-1]
es = np.log10(es)

plt.scatter(nb,es)
plt.show()
