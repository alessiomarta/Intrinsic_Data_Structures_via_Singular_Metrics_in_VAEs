from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_2dimg(img,fname):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig(fname)

#Generate the images for the application to an inverse problem section
def generate_2dimages(mix_vae,num_generators,latent_dims,dataloader,device,n_samples=256):
    with torch.no_grad():
        sample_inputs, _ = next(iter(dataloader))
        fixed_input = sample_inputs[0:32, :, :, :]
        
        #Original images of the last batch of the dataset
        img = make_grid(fixed_input, nrow=8, padding=2, normalize=False,
                        scale_each=False, pad_value=0)
        plt.figure()
        save_2dimg(img,"original.png")
        plt.clf()
        plt.close()

        #For each generator of the mixture of VAEs plot the reconstruction and sample some images
        for i in range(num_generators):
            with torch.no_grad():
                fixed_input = fixed_input.to(device)
                #Reconstructed images
                recon_batch, _, _, _ = mix_vae.decoders[i](fixed_input)
                recon_batch = recon_batch.cpu()
                recon_batch = make_grid(recon_batch, nrow=8, padding=2, normalize=False,
                                        scale_each=False, pad_value=0)
                plt.figure()
                save_2dimg(recon_batch,"rec"+str(i)+".png")
                plt.clf()
                plt.close()
            
            #Sample images from latent space
            z = torch.randn(n_samples,latent_dims).to(device)
            with torch.no_grad():
                samples = mix_vae.decoders[i].decode(z)
                samples = samples.cpu()
                samples = make_grid(samples, nrow=16, padding=2, normalize=False,
                                        scale_each=False, pad_value=0)
                plt.figure(figsize = (8,8))
                save_2dimg(samples,str(i)+"_sample.png")
                plt.clf()
                plt.close()

#Generate the images for the circle example
def generate_circle_images(mix_vae,num_generators,dataloader,device,n_samples=64):
    with torch.no_grad():

        sample_inputs, _ = next(iter(dataloader))
        fixed_input = sample_inputs[0:n_samples]
        pts = fixed_input.detach().cpu().numpy()
        plt.figure()
        plt.scatter(pts[:,0],pts[:,1])
        plt.savefig("original.png")
        plt.clf()
        plt.close()

        for i in range(num_generators):
            with torch.no_grad():
                fixed_input = fixed_input.to(device)
                recon_batch, _, _, _ = mix_vae.decoders[i](fixed_input)
                recon_batch = recon_batch.cpu()

                plt.figure()
                plt.scatter(pts[:,0],pts[:,1])
                plt.savefig("reconstruction.png")
                plt.clf()
                plt.close()

#Generate the 3d plots for the paraboloid examples
def generate_paraboloid_images(mix_vae,num_generators,device):
    with torch.no_grad():
        
        #Generate the paraboloid
        nx, ny = (10, 10)
        x = np.linspace(-.9, .9, nx, dtype=np.float32)
        y = np.linspace(-.9, .9, ny, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        xycoords = np.array((xv.ravel(), yv.ravel())).T
        z = .5 * (xycoords[:,0]**2+xycoords[:,1]**2)
        coords = np.zeros((nx*ny,3))
        coords[:,:2] = xycoords
        coords[:,2] = z

        fixed_input = torch.tensor(coords,dtype=torch.float32).to(device)
        
        #Plot the original paraboloid
        pts = fixed_input.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2], linewidth=0, antialiased=True)
        plt.savefig("original.png")
        plt.clf()
        plt.close()

        for i in range(num_generators):
            with torch.no_grad():
                #Recosntruct the paraboloid
                fixed_input = fixed_input.to(device)
                recon_batch, _, _, _ = mix_vae.decoders[i](fixed_input)
                recon_batch = recon_batch.cpu()

                #Plot the reconstructed paraboloid
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot_trisurf(recon_batch[:,0], recon_batch[:,1], recon_batch[:,2])
                plt.savefig("reconstruction.png")
                plt.clf()
                plt.close()