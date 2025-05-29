import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------

class SimpleVAE(nn.Module):
    """
    Model of a simple linear VAE for the circle and the paraboloid examples
    """
    def __init__(self, in_features = 3, latent_features = 3, beta = 1):

        """

        Args:
            in_features  dimensionality of the input.
            latent_features: dimensionality of the latent space.
            beta: weight of the KL divergence in the loss function.

        """
        super(SimpleVAE, self).__init__()

        self.latent_features = latent_features
        self.in_features = in_features
        self.beta = beta
 
        #Encoder
        self.enc1 = nn.Linear(in_features=self.in_features, out_features=4*self.in_features)
        self.enc2 = nn.Linear(in_features=4*self.in_features, out_features=4*self.in_features)
        self.enc3 = nn.Linear(in_features=4*self.in_features, out_features=self.latent_features*2)
 
        #Decoder
        self.dec1 = nn.Linear(in_features=self.latent_features, out_features=4*self.in_features)
        self.dec2 = nn.Linear(in_features=4*self.in_features, out_features=4*self.in_features)
        self.dec3 = nn.Linear(in_features=4*self.in_features, out_features=self.in_features)

        #Xavier initialization
        nn.init.xavier_uniform_(self.enc1.weight) 
        nn.init.xavier_uniform_(self.enc2.weight) 
        nn.init.xavier_uniform_(self.enc3.weight) 
        nn.init.xavier_uniform_(self.dec1.weight) 
        nn.init.xavier_uniform_(self.dec2.weight) 
        nn.init.xavier_uniform_(self.dec3.weight) 


    def reparameterize(self, mu, log_var):

        #Compute the standard deviation
        std = torch.exp(0.5*log_var)
        
        #Generate random numbers with randn_like, as we need the same size
        eps = torch.randn_like(std) 
        
        #Sampling as if coming from the input space
        sample = mu + (eps * std)

        return sample
 
    def forward(self, x):

        #Encoding
        y = F.relu(self.enc1(x))
        y = F.relu(self.enc2(y))
        y = self.enc3(y).view(-1, 2, self.latent_features)
        
        #Get mu and log_var: first feature values as mean and second one as variance
        mu = y[:, 0, :]
        log_var = y[:, 1, :]
        
        #Get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        #Decoding
        y = F.relu(self.dec1(z))
        y = F.relu(self.dec2(y))
        reconstruction = 2*torch.sigmoid(self.dec3(y))-1.
        return reconstruction, x, mu, log_var
    
    def reduce(self,x):
        
        #Encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x).view(-1, 2, self.latent_features)
        
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        
        #Get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        return z

    
    def negative_ELBO(self, x):
        recon_x, _, mu, logvar = self.forward(x)

        ediff =  torch.abs(recon_x-x)**2
        ediff = ediff.reshape(ediff.shape[0],-1)
        recon_loss = torch.sqrt(torch.sum(ediff,dim=1))

        kl_divergence = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1)
        loss = recon_loss + self.beta * kl_divergence

        return loss, recon_loss, kl_divergence

#----------------------------------------------------------------------------------
    
class CNNVAE(nn.Module):
    def __init__(self, latent_dims, beta):

        """ 
        Convolution beta-VAE model for the "Application to an inverse problem" section.

        Args:

            latent_features: dimensionality of the latent space.
            beta: weight of the KL divergence in the loss function.
        """
        super(CNNVAE, self).__init__()
        self.latent_dims = latent_dims
        self.beta = beta


        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 8,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Conv2d(
                in_channels = 8,
                out_channels = 8,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 8,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Conv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
        )
        
        
        nn.init.xavier_uniform_(self.conv_1[0].weight) 
        nn.init.xavier_uniform_(self.conv_2[0].weight) 

        self.fc_enc = nn.Sequential(
            nn.Linear(16*16*64, 64 * 4), nn.ReLU(),
            nn.Linear(4 * 64, 64 * 2), nn.ReLU(),
            )
        
        nn.init.xavier_uniform_(self.fc_enc[0].weight)
        nn.init.xavier_uniform_(self.fc_enc[2].weight)

        self.fc_mu = nn.Linear(64 * 2, self.latent_dims)
        self.fc_var = nn.Linear(64 *2, self.latent_dims)

        nn.init.xavier_uniform_(self.fc_mu.weight) 
        nn.init.xavier_uniform_(self.fc_var.weight) 

        
        self.fc_dec = nn.Sequential(
            nn.Linear(self.latent_dims,64 * 2), nn.ReLU(),
            nn.Linear(64 * 2,64 * 4), nn.ReLU(),
            nn.Linear(64 * 4,16*16*64), nn.ReLU(),
            )
        
        nn.init.xavier_uniform_(self.fc_dec[0].weight)
        nn.init.xavier_uniform_(self.fc_dec[2].weight)
        nn.init.xavier_uniform_(self.fc_dec[4].weight)

        self.tconv_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding= 1
            ),
            nn.ConvTranspose2d(
                in_channels = 64,
                out_channels = 32,
                kernel_size = 2,
                stride = 2
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
        )

        self.tconv_2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 32,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding= 1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
        )

        self.tconv_3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 2,
                stride = 2,
            ),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
        )

        self.tconv_4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 16,
                out_channels = 8,
                kernel_size = 3,
                stride = 1,
                padding= 1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
        )

        self.tconv_5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 8,
                out_channels = 8,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ),
            nn.Conv2d(
                in_channels = 8,
                out_channels = 1,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc_enc(out)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return mu, log_var

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        out = self.fc_dec(z)
        out = out.view(z.shape[0], 64, 16, 16)
        out = self.tconv_1(out)
        out = self.tconv_2(out)
        out = self.tconv_3(out)
        out = self.tconv_4(out)
        out = self.tconv_5(out)
        return out

    def forward(self, x):
        
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return self.decode(z), x, mu, log_var

    def reduce(self,x):
        mu, lv = self.encode(x)
        out = self.reparametrize(mu,lv)
        return out      
    
    
    def negative_ELBO(self, x):
        recon_x, _, mu, logvar = self.forward(x)

        #Compute tanimoto distance
        ediff =  torch.abs(recon_x-x)
        ediff = ediff.reshape(ediff.shape[0],-1)
        r_f = recon_x.reshape(recon_x.shape[0],-1)
        x_f = x.reshape(x.shape[0],-1)
        dot = (r_f*x_f).sum(axis = 1)
        rf2 = (r_f**2).sum(axis = 1)
        xf2 = (x_f**2).sum(axis = 1)
        tanimoto = 1. - dot/(rf2+xf2-dot)   
        recon_loss = 128 * 128 * tanimoto

        #KL divergence
        kl_divergence = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1)

        #Lasso ragularization
        lasso =  sum([p.abs().sum() for p in self.parameters()])

        loss = recon_loss + self.beta*kl_divergence + 0.0001 * lasso
        return loss, recon_loss, kl_divergence
    
 #----------------------------------------------------------------------------------
    
class LatentNet(nn.Module):
    """
    Model for the linear VAE employed to transform the latent representation of the images 
    into the latent representation of their sinograms and vice-versa
    """
    def __init__(self, in_features):
        super(LatentNet, self).__init__()

        self.in_features = in_features
 
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=2*in_features)
        self.fc2 = nn.Linear(in_features=2*self.in_features, out_features=2*in_features)

        self.bn1 = nn.BatchNorm1d(2*in_features)

        self.fc3 = nn.Linear(in_features=2*self.in_features, out_features=4*in_features)
        self.fc4 = nn.Linear(in_features=4*self.in_features, out_features=8*in_features)
        self.fc5 = nn.Linear(in_features=8*self.in_features, out_features=8*in_features)
        self.fc6 = nn.Linear(in_features=8*in_features, out_features=4*in_features)
        self.fc7 = nn.Linear(in_features=4*in_features, out_features=2*in_features)
        
        self.bn2 = nn.BatchNorm1d(2*in_features)
        
        self.fc8 = nn.Linear(in_features=2*self.in_features, out_features=2*in_features)
        self.fc9 = nn.Linear(in_features=2*self.in_features, out_features=in_features)

        
 
    def forward(self, x):       
        x =  F.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        x = self.bn1(x)
        
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.bn2(x)
    
        x =  F.relu(self.fc8(x))
        x =  F.sigmoid(self.fc9(x))
        
        return x
        
    def loss(self, xs, ys):
        
        recon = self.forward(xs)
        diff =  torch.abs(recon-ys)**2
        loss = torch.sqrt(torch.sum(diff,dim=0))
        loss = torch.sum(loss)/len(xs)
        
        return loss