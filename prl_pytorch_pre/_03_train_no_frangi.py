#bsub -q lpcgpu -gpu "num=1" -n 1 -J "run_autoencoder[1-10]" -o ~/Documents/prl/stdout/autoencoder.txt bash -c "python /home/fengling/Documents/prl/prl_pytorch/_03_train.py"
#bsub -q taki_normal -n 21 -J "pretrain[1-10]" bash -c "python /home/fengling/Documents/prl/prl_pytorch/_03_train.py"
import os
from random import randrange
import torch
import torch.nn as nn
import torchvision as tv
import torchio as tio
import _01_dataloader_no_frangi as prl_dl
import _02_autoencoder_no_frangi as prl_ae

#import importlib
#importlib.reload(prl_dl)
#importlib.reload(prl)

def isolate_lesion(patches_batch):
    lesion_mask_tensor = patches_batch["lesion_mask"]["data"]
    mask_tensor = torch.zeros_like(lesion_mask_tensor)
    for i in range(lesion_mask_tensor.size()[0]):
        tmp_lesion_mask = lesion_mask_tensor[i, :, :, :, :].clone()
        lesion_ids = tmp_lesion_mask.unique()
        lesion_ids = lesion_ids[lesion_ids != 0]

        if (len(lesion_ids) > 1): # Only need to grey out lesion if there are two lesions.
            id_to_keep = lesion_ids[randrange(len(lesion_ids))]
            mask_tensor[i, :, :, :, :] = 0.1 + 0.9 * ((tmp_lesion_mask == 0) + (tmp_lesion_mask == id_to_keep))
        else:
            mask_tensor[i, :, :, :, :] = torch.ones_like(tmp_lesion_mask)
            
    return(mask_tensor.repeat(1, len(keys), 1, 1, 1))

def make_weight_tensor(patches_batch, isolation_mask, lesion_weight, epi_phase_weight):
    in_lesion = (patches_batch["lesion_mask"]["data"] != 0) * (isolation_mask[:, 0:1, :, :, :] == 1)
    weight_tensor = in_lesion * (lesion_weight - 1) + 1
    weight_tensor = weight_tensor.repeat(1, len(keys), 1, 1, 1)
    weight_tensor[:, 2:4, :, :, :] = weight_tensor[:, 2:4, :, :, :] * epi_phase_weight
    
    return(weight_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

prl_autoencoder = prl_ae.Autoencoder3D()
prl_autoencoder = prl_autoencoder.to(device)
optimizer = torch.optim.Adam(prl_autoencoder.parameters(), lr=0.001)

num_epochs = 50
keys = ["t1", "flair", "epi", "phase"]
for epoch_index in range(num_epochs):
    epoch_loss = 0
    loss_fn = nn.MSELoss()
        
    for patches_batch in prl_dl.train_loader:
        optimizer.zero_grad()
        isolation_mask_tensor = isolate_lesion(patches_batch)
        input_tensor = torch.cat([patches_batch.get(key)["data"] for key in keys], dim=1) * isolation_mask_tensor
        input_tensor = input_tensor.to(device)
        output_tensor = prl_autoencoder(input_tensor)
        
        weight_tensor = make_weight_tensor(patches_batch, isolation_mask_tensor, 10, 5) # Upweight reconstruction of epi and phase by 10:1. Upweight reconstruction of primary lesion by 5:1
        weight_tensor = weight_tensor.to(device)
        loss = loss_fn(output_tensor * weight_tensor, input_tensor * weight_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
    print("Batch " + str(epoch_index) + ": Loss = " + str(epoch_loss)) 

torch.save(prl_autoencoder.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_models/prl_autoencoder_05_" + str(prl_dl.i) + ".pt")
