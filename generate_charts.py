import torch
import os
import utils.sampling
from utils.injective_generators import InjectiveDenseGenerator
from utils.mix_vae_invertible import Mix_VAE
import numpy as np
import torch.nn as nn
from utils.datasets import ImagesDataset, Jittering

from math import sqrt
import matplotlib.pyplot as plt


#Set the device
GPU = True
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')


path_charts = "local_charts"
isExist = os.path.exists(path_charts)
if not isExist:
   os.makedirs(path_charts)

model = torch.load("img_0_model.pth").to(device)


n_data = 40000
n_test_data= 5000
batch_size = 256

#Generate data

train_dataset=ImagesDataset(epoch_size=n_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),eps=6,fix_int=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

test_dataset=ImagesDataset(epoch_size=n_test_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),test=True,eps=6,fix_int=True)    
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

#Encode data in Whitney VAE

encoded_train_data = []
encoded_test_data = []

enc = []

print("Encoding images...")
counter = 0
with torch.no_grad():
    for xs,ids in train_dataloader:

        print(counter)
        counter += len(xs)
        xs = xs.to(device) 
        ys = model.reduce(xs.detach())
        
        tmp = [(a,b) for (a,b) in zip(ys,ids)]

        enc.extend(ys)
        encoded_train_data.extend(tmp)
    
    wtrain_dataloader=torch.utils.data.DataLoader(encoded_train_data,batch_size=batch_size,shuffle=True,drop_last=True)

    for xs,ids in test_dataloader:
        xs = xs.to(device)
        ys = model.reduce(xs.detach())
        tmp = [(a,b) for (a,b) in zip(ys,ids)]
        encoded_test_data.extend(tmp)
    
    wtest_dataloader=torch.utils.data.DataLoader(encoded_test_data,batch_size=batch_size,shuffle=True,drop_last=True)


#Set the ID and the number of charts, then build the local charts

num_generators = 2
dim_hd = 12*2 + 1
dim_ld = 12
sig_ld=0.01
sig_hd=.1
learning_rate = 5e-3
n_epochs = 100

decoders = []

for ng in range(num_generators):
    decoders.append(InjectiveDenseGenerator(dim_hd,dim_ld,sig_hd,sig_ld,utils.sampling.get_spline_latent,latent_nf=True,num_layers=5,sub_net_size=dim_hd*2,latent_NF_layers=3).to(device))

charts_vae = Mix_VAE(decoders)
optimizer=torch.optim.Adam(charts_vae.parameters(), lr = learning_rate)

#Plot functions

def show_side_coord(lt_img,name,folder):
    dims = int(sqrt(lt_img.shape[0]))
    plt.axis('off')
    _, axs = plt.subplots(dims, dims, figsize=(12, 12))
    axs = axs.flatten()
    lt_img=lt_img.detach().cpu().numpy()
    for img, ax in zip(lt_img, axs):
        ax.set_axis_off()
        ax.axis("off")
        ax.imshow(img[0],cmap='gist_gray')
    plt.axis('off')
    plt.savefig(folder+"/"+name+'.png')
    plt.close()

def chart_2d_img(generator,n_samples,name,folder,axis1=0,axis2=0):
    ng = generator
    with torch.no_grad():
        X = np.arange(charts_vae.relevant_region[0],charts_vae.relevant_region[1], (charts_vae.relevant_region[1]-charts_vae.relevant_region[0])/n_samples)
        local_coords=torch.zeros((len(X)*len(X),charts_vae.decoders[0].dim_ld)).to(device)
        for i,x in enumerate(X):
            local_coords[i*len(X):(i+1)*len(X),axis1] = torch.tensor(x).item()
            for j,y in enumerate(X):
                local_coords[i*len(X)+j,axis2] = torch.tensor(y).item()

        samples=charts_vae.decoders[ng](local_coords)
        samples = model.decode(samples)   
        xs_gen=torch.clamp(samples,min=0,max=1).detach().cpu().numpy()

    show_side_coord(samples,name,folder)
    return xs_gen


print("Training local charts VAEs...")

reseed = True

if reseed:
    if num_generators>1:
        # seeding does not make sense for one generator.
        charts_vae.seeding(wtrain_dataloader,num_samples=400,init_epochs=250,seeding_candidates=batch_size,batch_size=200,learning_rate=1e-2)
    # compute test loss and save initialization
    loss_sum=charts_vae.test_step(wtest_dataloader)
    best_test_loss=loss_sum
    print('After seeding, Test loss: {0:.2f}, Best test loss: {1:.2f}, sig_ld: {2:.4f}'.format(loss_sum,best_test_loss,sig_ld),flush=True)

optimizer=torch.optim.Adam(charts_vae.parameters(), lr = learning_rate)


for epoch in range(n_epochs):

    print("Epoch:",epoch)
    lip_loss = 0.01
    train_loss=charts_vae.train_epochs_full_grad(wtrain_dataloader,optimizer,gradient_clipping=2.,Lipschitz_loss=lip_loss)
    optimizer.zero_grad()
    
    if epoch % 10:
        charts_vae.save_checkpoint('chart_gen_tmp.pt')
        smp = charts_vae.decoders[0].sample(100)
        imgs = model.decode(smp)
        show_side_coord(imgs,"tmp_chart",path_charts)
    

charts_vae.save_checkpoint('chart_gen_tmp.pt')
smp = charts_vae.decoders[0].sample(100)

with torch.no_grad():
    for gn in range(num_generators):
        for i in range(dim_ld):
            for j in range(dim_ld):
                if (i != j):
                    fname = str(gn)+"_"+str(i)+"_"+str(j)
                    chart_2d_img(gn,10,fname,path_charts,i,j)