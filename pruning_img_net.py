import torch
import matplotlib.pyplot as plt
import torch_pruning as tp
import torchvision.transforms as transforms
import numpy as np
from utils.datasets import ImagesDataset, Jittering
from torchsummary import summary

#Set the device
GPU = True
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')


transform = transforms.Compose([transforms.ToTensor(),])

n_data = 5000
n_test_data = 10
batch_size = 512

#Generate dataset
train_dataset=ImagesDataset(epoch_size=n_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),eps=6,fix_int=True)
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False,drop_last=False)

test_dataset=ImagesDataset(epoch_size=n_test_data,start_seed=0,std_intensity=.5,intensity_dist='uniform',transform=Jittering(),test=True,eps=6,fix_int=True)    
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)

prs = np.arange(0.,0.95,0.05)

for pr in prs:

    model_img = torch.load("img_0_model.pth").to(device)
    model = model_img.eval()
    example_inputs = torch.randn(1, 1, 128, 128).to(device)

    #Importance criterion
    impcr = tp.importance.GroupNormImportance(p=1,normalizer=None,group_reduction="mean") 

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []

    for n,m in model.named_modules():
        #Do not prune the first layer
        if isinstance(m, torch.nn.Conv2d) and m.in_channels == 1:
            ignored_layers.append(m) # DO NOT prune the final layer!
        #Do not prune the final layer
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == 1:
            ignored_layers.append(m)
        #Do not prune the last layer of the encoder
        if n == "fc_mu" or n == "fc_var":
            ignored_layers.append(m)

    iterative_steps = 1

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=impcr,
        pruning_ratio=pr,
        ignored_layers=ignored_layers,
        round_to=1,
        global_pruning=True,
        iterative_steps=iterative_steps,
        max_pruning_ratio=1.
    )

    #Prune the model
    for i in range(iterative_steps):
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")


    #Print summary of the model

    INPSIZE= (1, 128, 128)
    print("Pruned model summary")
    summary(model,INPSIZE)
    model_original = torch.load("img_0_model.pth").to(device)
    print("Original model summary")
    summary(model_original,INPSIZE)

    #Save pruned model
    spr = f"{pr:.2f}"
    model.zero_grad()
    torch.save(model,str(spr)+'_img_model_pruned.pth')


    #Plot a reconstruction image wiht the original and pruned network
    example_inputs = [a.to(device) for a,b in test_dataloader]
    example_inputs = torch.vstack(example_inputs)
    example_inputs = example_inputs[0][None,:,:,:]

    original = model_original(example_inputs)[0]
    pruned = model(example_inputs)[0]

    img=original.detach().cpu().numpy()
    plt.imshow(img[0,0],cmap='gist_gray')
    plt.savefig(str(spr)+"_original.png")
    plt.clf()

    img=pruned.detach().cpu().numpy()
    plt.imshow(img[0,0],cmap='gist_gray')
    plt.savefig(str(spr)+"_pruned.png")
    plt.clf()
