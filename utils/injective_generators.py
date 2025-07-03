#!/usr/bin/env python

import torch.nn as nn
import torch
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock,ActNorm, PermuteRandom
import numpy as np
from utils.general import get_device

device = get_device()

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)

def subnet_fc(c_in, c_out,sub_net_size=64):
    net = nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                         nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                         nn.Linear(sub_net_size,  c_out))
    net.apply(init_weights)
    net[-1].weight = nn.Parameter(torch.zeros_like(net[-1].weight))
    return net

def DenseINN(input_shape,num_layers=5,sub_net_size=64,actnorm=False):
        
        nodes = [InputNode(input_shape, name='input')]
        for k in range(num_layers):
            if actnorm:
                nodes.append(Node(nodes[-1], ActNorm, {}))
            nodes.append(Node(nodes[-1],
                            GLOWCouplingBlock,
                            {'subnet_constructor':lambda c_in,c_out: subnet_fc(c_in,c_out,sub_net_size=sub_net_size), 'clamp':1.4},
                            name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1], PermuteRandom, {'seed':k}, name=F'permute_dense_res_{k}'))
        nodes.append(OutputNode(nodes[-1], name='output'))

        return ReversibleGraphNet(nodes, verbose=False).to(device)


class InjectiveGenerator(nn.Module):
    def __init__(self,dim_ld,sig_hd,sig_ld,get_latent,latent_nf=False,rev=False,latent_NF_layers=3,inn_size=64,beta=1.):
        super(InjectiveGenerator,self).__init__()
        self.sample_latent,self.latent_energy,self.relevant_region=get_latent()
        self.dim_ld=dim_ld
        self.rev=rev
        self.sig_hd=nn.Parameter(torch.tensor(sig_hd,dtype=torch.float,device=device),requires_grad=False)
        self.sig_ld_log=nn.Parameter(torch.log(torch.tensor(sig_ld,dtype=torch.float,device=device)),requires_grad=False)
        self.beta=nn.Parameter(torch.tensor(beta,dtype=torch.float,device=device),requires_grad=True)
        self.latent_nf=latent_nf
        self.VAE_logdet=False
        if self.latent_nf:
            self.latent_INN = DenseINN(self.dim_ld,latent_NF_layers,inn_size)
            self.add_module("latent_INN",self.latent_INN)       

    def forward_VAE(self,x):
        raise NotImplementedError('Forward evaluation of the injective generator has to be specified!')

    def backward_VAE(self,x):
        raise NotImplementedError('Backward evaluation of the injective generator has to be specified!')

    def forward(self,x):        
        if self.latent_nf:
            x,ld=self.latent_INN(x,rev=self.rev)
        out=self.forward_VAE(x)
        return out

    def backward(self,x):
        out=self.backward_VAE(x)
        if self.latent_nf:
            out,ld=self.latent_INN(out,rev= not self.rev)
        return out

    def set_sig_ld(self,sig_ld):
        if sig_ld>0:
            self.sig_ld_log.data=torch.log(torch.tensor(sig_ld,dtype=torch.float,device=device))
        else:
            self.sig_ld_log.requires_grad_(True)

    def set_sig_hd(self,sig_hd):
        self.sig_hd.data=torch.tensor(sig_hd,dtype=torch.float,device=device)

   
    def get_sig_ld(self):
        return torch.exp(self.sig_ld_log)

    def get_std(self,x):
        return torch.exp(self.sig_ld_log)

    def negative_ELBO(self,xs,scale=None,return_latent_probs=False):
        self.VAE_logdet = True
        zs=self.backward_VAE(xs)
        stds=self.get_std(xs)
        
        if not self.VAE_logdet or not self.latent_nf:
            zs=zs+stds*torch.randn_like(zs)
            xs_recon=self.forward_VAE(zs)
            error= xs-xs_recon
            error=torch.reshape(error,[error.shape[0],-1])     
            logdet = .5*torch.sum((error/self.sig_hd)**2,dim=-1)
            logdet-=torch.sum(torch.log(stds),-1)
        if self.latent_nf:
            zs_latent,ld=self.latent_INN(zs,rev=not self.rev)
            if self.VAE_logdet:
                zs_latent=zs_latent+stds*torch.randn_like(zs)
                zs_recon,_=self.latent_INN(zs_latent,rev=self.rev)
                xs_recon=self.forward_VAE(zs_recon)
                
                error=xs_recon-xs
                error=torch.reshape(error,[error.shape[0],-1])
                logdet = .5*torch.sum((error/self.sig_hd)**2,dim=-1)
            else:
                logdet-=ld
        else:
            zs_latent=zs
        logpz=self.latent_energy(zs_latent)
        loss=logpz
        torch.cuda.empty_cache()

        if not scale is None:        
            loss_scale=(self.latent_energy(zs_latent*scale))
            loss_add=(loss_scale-loss).squeeze()

        loss= loss+ logdet

        if scale is None:
            if return_latent_probs:
                return loss,logpz
            return loss
        else:
            if return_latent_probs:
                return loss,loss_add,logpz
            return loss,loss_add


    def sample(self,n):
        zs=self.sample_latent((n,self.dim_ld))
        samples=self.forward(zs)
        return samples

    def Lipschitz_loss(self,batch_size):
        zs=self.sample_latent((batch_size,self.dim_ld))
        zs2=zs+self.get_sig_ld()*torch.randn_like(zs)
        xs=self.forward(zs)
        xs2=self.forward(zs2)

        torch.cuda.empty_cache()
        
        return torch.sum(((xs-xs2)/self.sig_hd)**2)
    


class InjectiveDenseGenerator(InjectiveGenerator):
    def __init__(self,dim_hd,dim_ld,sig_hd,sig_ld,get_latent,num_layers=5,sub_net_size=64,INN_constructor=DenseINN,latent_nf=False,rev=False,latent_NF_layers=3,inn_size=64):
        super(InjectiveDenseGenerator,self).__init__(dim_ld,sig_hd,sig_ld,get_latent,latent_nf=latent_nf,rev=rev,latent_NF_layers=latent_NF_layers,inn_size=inn_size)
        self.sample_latent,self.latent_energy,self.relevant_region=get_latent()
        self.dim_hd=dim_hd
        self.INN = INN_constructor(self.dim_hd,num_layers,sub_net_size,actnorm=self.rev).to(device)
        self.add_module("INN",self.INN)  

    def forward_VAE(self,x):
        out=torch.zeros((x.shape[0],self.dim_hd),device=device)
        out[:,0:self.dim_ld]=x
        out=self.INN(out,rev=self.rev)[0]
        return out

    def backward_VAE(self,x):
        out = x
        out=self.INN(out,rev=not self.rev)[0]
        out=out[:,0:self.dim_ld]

        return out
