#!/usr/bin/env python

from torch.utils.data import Dataset
import torch
import generate_data

class ImagesDataset(Dataset):
    def __init__(self, epoch_size=1,transform=None,start_seed=None,std_intensity=0.1,test=False,intensity_dist='gauss',eps=4,fix_int=False,device=None,n_balls=2):     
        self.len=epoch_size
        self.transform=transform
        self.n_balls=n_balls
        self.std_intensity=std_intensity
        self.test=test
        self.intensity_dist=intensity_dist
        self.eps=eps
        self.fix_int=fix_int
        self.images = []
        
        for i in range(self.len):
            if (device == None):
                    image = torch.tensor(generate_data.generate_image(self.n_balls,std_intensity=self.std_intensity,intensity_dist=self.intensity_dist,eps=self.eps,fix_int=self.fix_int)[:,None],dtype=torch.float)
            else:
                image = torch.tensor(generate_data.generate_image(self.n_balls,std_intensity=self.std_intensity,intensity_dist=self.intensity_dist,eps=self.eps,fix_int=self.fix_int)[:,None,:],dtype=torch.float,device=device)
            self.images.append(image)

        
        self.images  = torch.stack(self.images).swapaxes(1,2) 

                    
    def __len__(self):
        return self.len

    def reshuffle(self):
        self.start_seed+=self.len

    def __getitem__(self, idx):
        
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)
        
        return img,idx
    


class ParaboloidDataset(Dataset):
    def __init__(self, epoch_size=1, device=None):     
        self.data = []
        self.len=epoch_size
        
        for i in range(self.len):
            if (device == None):
                data = torch.tensor(generate_data.gen_paraboloid())
            else:
                data = torch.tensor(generate_data.gen_paraboloid()).to(device)
            
            self.data.append(data)
        
        self.data  = torch.stack(self.data) 

    def __len__(self):
        return self.len

    def reshuffle(self):
        self.start_seed+=self.len

    def __getitem__(self, idx):
        
        img = self.data[idx]
        
        return img,idx


class CircleDataset(Dataset):
    def __init__(self, epoch_size=1, device=None):     
        self.data = []
        self.len=epoch_size
        
        for i in range(self.len):
            if (device == None):
                data = torch.tensor(generate_data.gen_circle())
            else:
                data = torch.tensor(generate_data.gen_circle()).to(device)
            
            self.data.append(data)
        
        self.data  = torch.stack(self.data) 

    def __len__(self):
        return self.len

    def reshuffle(self):
        self.start_seed+=self.len

    def __getitem__(self, idx):
        
        img = self.data[idx]
        
        return img,idx   


class Jittering(object):
    def __call__(self, tensor):
        return (255*tensor + torch.rand_like(tensor))/256

    def __repr__(self):
        return self.__class__.__name__
