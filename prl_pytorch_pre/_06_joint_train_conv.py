#bsub -q taki_normal -n 21 -J "train_predictor[1-10]" bash -c "python /home/fengling/Documents/prl/prl_pytorch_pre/_06_joint_train_conv.py"
import os
from random import randrange
import torch
import torch.nn as nn
import torchvision as tv
import torchio as tio
import _04_predictor_dataloader_no_frangi as prl_dl
import _02_autoencoder_conv as prl_ae
import _05_predictor_conv as prl_pred

#import importlib
#importlib.reload(prl_dl)
#importlib.reload(prl_ae)
#importlib.reload(prl_pred)

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

def get_lesion_type(patches_batch, isolation_mask_tensor):
    lesion_type_tensor = patches_batch["lesion_type"]["data"][:, 0, :, :, :]
    isolation_mask = (isolation_mask_tensor[:, 0, :, :, :] - 0.1) * 10 / 9 # Originally is a [batch, 4, 24, 24, 24] tensor
    isolated_lesion_type = lesion_type_tensor * isolation_mask

    batch_size = isolated_lesion_type.size()[0]
    target_tensor = torch.zeros(batch_size, 3)
    weight_tensor = torch.zeros(batch_size, 3)

    for i in range(batch_size):
        tmp_unique = isolated_lesion_type[i, :, :, :].unique()
        tmp_unique = int(tmp_unique[len(tmp_unique) - 1].item())
        target_tensor[i, :], weight_tensor[i, :] = process_lesion_type(tmp_unique, 
                                                                       patches_batch["contains_lesions"][i], 
                                                                       patches_batch["contains_cvs"][i])
    return([target_tensor, weight_tensor])

def process_lesion_type(lesion_id, contains_lesions, contains_cvs): # Return tensor of [is_lesion, is_PRL, is_CVS]
    target = torch.zeros(3)
    weight = torch.ones(3)
    weight[1] = 1
    weight[2] = 1
    
    if contains_lesions == False:
        weight[0] = 0
    if contains_cvs == False:
        weight[2] = 0
   
    digits = [int(x) for x in str(lesion_id)] 
    # First digit is always 1 for computational convenience
   
    if (lesion_id == 0):
        return([target, weight]) # Just return the tensor of 0 for target and normal weights
    
    if digits[1] == 1: # standard lesion
        weight[0] = 1
        target[0] = 1
        
    if digits[2] == 1: # PRL lesion
        weight[0] = 1
        target[0] = 1
        weight[1] = 1
        target[1] = 1
    
    if digits[3] == 1: # CVS lesion
        weight[0] = 1
        target[0] = 1
        weight[2] = 1
        target[2] = 1
    
    return([target, weight])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

prl_autoencoder = prl_ae.Autoencoder3D()
prl_autoencoder = prl_autoencoder.to(device)

model_path = "/home/fengling/Documents/prl/prl_pytorch_pre/cv_models_conv/prl_autoencoder_05_" + str(prl_dl.i) + ".pt"
 
# Load the saved model
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Load the state_dict from the checkpoint
prl_autoencoder.load_state_dict(checkpoint)

prl_predictor = prl_pred.Predictor3D()
prl_predictor = prl_predictor.to(device)

predictor_optimizer = torch.optim.Adam(prl_predictor.parameters(), lr=0.001)
joint_optimizer = torch.optim.Adam(list(prl_autoencoder.encoder.parameters()) + 
                                   list(prl_predictor.parameters()), 
                                   lr=0.001)

num_epochs = 50
keys = ["t1", "flair", "epi", "phase"]

for epoch_index in range(num_epochs):
    epoch_loss = 0
    
    # First train the predictor to catch up to pre-trained encoder
    if epoch_index <= num_epochs * 0.1:
        for patches_batch in prl_dl.train_loader:
            predictor_optimizer.zero_grad()

            isolation_mask_tensor = isolate_lesion(patches_batch)
            target_tensor, weight_tensor = get_lesion_type(patches_batch, isolation_mask_tensor)
            target_tensor = target_tensor.to(device)
            weight_tensor = weight_tensor.to(device)

            input_tensor = torch.cat([patches_batch.get(key)["data"] for key in keys], dim=1) * isolation_mask_tensor
            input_tensor = input_tensor.to(device)

            encoded_tensor = prl_autoencoder.get_latent(input_tensor)
            output_tensor = prl_predictor(encoded_tensor)

            loss = nn.BCELoss()(output_tensor * weight_tensor, target_tensor * weight_tensor)
            loss.backward()
            predictor_optimizer.step()

            epoch_loss += float(loss)
        print("Epoch " + str(epoch_index) + ": Loss = " + str(epoch_loss))
        continue
    
    # Otherwise, train both predictor and encoder together
    for patches_batch in prl_dl.train_loader:
        joint_optimizer.zero_grad()
        
        isolation_mask_tensor = isolate_lesion(patches_batch)
        target_tensor, weight_tensor = get_lesion_type(patches_batch, isolation_mask_tensor)
        target_tensor = target_tensor.to(device)
        weight_tensor = weight_tensor.to(device)
        
        input_tensor = torch.cat([patches_batch.get(key)["data"] for key in keys], dim=1) * isolation_mask_tensor
        input_tensor = input_tensor.to(device)
        
        encoded_tensor = prl_autoencoder.get_latent(input_tensor)
        output_tensor = prl_predictor(encoded_tensor)
        
        loss = nn.BCELoss()(output_tensor * weight_tensor, target_tensor * weight_tensor)
        loss.backward()
        joint_optimizer.step()
           
        epoch_loss += float(loss)
    print("Epoch " + str(epoch_index) + ": Loss = " + str(epoch_loss)) 

torch.save(prl_autoencoder.state_dict(), "/home/fengling/Documents/prl/prl_pytorch_pre/cv_models_conv/prl_joint_05_" + str(prl_dl.i) + ".pt")
torch.save(prl_predictor.state_dict(), "/home/fengling/Documents/prl/prl_pytorch_pre/cv_models_conv/prl_predictor_05_" + str(prl_dl.i) + ".pt")
