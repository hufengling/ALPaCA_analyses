#bsub -q taki_normal -n 21 -J "train_solo[1-10]" bash -c "python /home/fengling/Documents/prl/prl_pytorch/_10_solo_train.py"
import os
from random import randrange
import torch
import torch.nn as nn
import torchvision as tv
import torchio as tio
import _04_predictor_dataloader as prl_dl
import _02_autoencoder as prl_ae
import _09_solo_predictor as prl_pred

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

        if (len(lesion_ids) > 1): # Only need to blackout lesion if there are two lesions.
            id_to_keep = lesion_ids[randrange(len(lesion_ids))]
            mask_tensor[i, :, :, :, :] = 0.1 * torch.ones_like(tmp_lesion_mask) + 0.9 * ((tmp_lesion_mask == 0) + (tmp_lesion_mask == id_to_keep))
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

lesion_autoencoder = prl_ae.Autoencoder3D()
prl_autoencoder = prl_ae.Autoencoder3D()
cvs_autoencoder = prl_ae.Autoencoder3D()
lesion_autoencoder = lesion_autoencoder.to(device)
prl_autoencoder = prl_autoencoder.to(device)
cvs_autoencoder = cvs_autoencoder.to(device)

model_path = "/home/fengling/Documents/prl/prl_pytorch/cv_models/prl_autoencoder_05_" + str(prl_dl.i) + ".pt"

# Load the saved model
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Load the state_dict from the checkpoint
lesion_autoencoder.load_state_dict(checkpoint)
prl_autoencoder.load_state_dict(checkpoint)
cvs_autoencoder.load_state_dict(checkpoint)

lesion_predictor = prl_pred.Predictor3D()
lesion_predictor = lesion_predictor.to(device)
prl_predictor = prl_pred.Predictor3D()
prl_predictor = prl_predictor.to(device)
cvs_predictor = prl_pred.Predictor3D()
cvs_predictor = cvs_predictor.to(device)

lesion_optimizer = torch.optim.Adam(lesion_predictor.parameters(), lr=0.001)
lesion_joint_optimizer = torch.optim.Adam(list(lesion_autoencoder.encoder.parameters()) + 
                                   list(lesion_predictor.parameters()), 
                                   lr=0.001)
prl_optimizer = torch.optim.Adam(prl_predictor.parameters(), lr=0.001)
prl_joint_optimizer = torch.optim.Adam(list(prl_autoencoder.encoder.parameters()) + 
                                   list(prl_predictor.parameters()), 
                                   lr=0.001)
cvs_optimizer = torch.optim.Adam(cvs_predictor.parameters(), lr=0.001)
cvs_joint_optimizer = torch.optim.Adam(list(cvs_autoencoder.encoder.parameters()) + 
                                   list(cvs_predictor.parameters()), 
                                   lr=0.001)

num_epochs = 50
keys = ["t1", "flair", "epi", "phase", "frangi"]

for epoch_index in range(num_epochs):
    epoch_loss = [0, 0, 0]
    
    # First train the predictor to catch up to pre-trained encoder
    if epoch_index <= num_epochs * 0.1:
        for patches_batch in prl_dl.train_loader:
            lesion_optimizer.zero_grad()
            prl_optimizer.zero_grad()
            cvs_optimizer.zero_grad()

            isolation_mask_tensor = isolate_lesion(patches_batch)
            target_tensor, weight_tensor = get_lesion_type(patches_batch, isolation_mask_tensor)
            target_tensor = target_tensor.to(device)
            weight_tensor = weight_tensor.to(device)

            input_tensor = torch.cat([patches_batch.get(key)["data"] for key in keys], dim=1) * isolation_mask_tensor
            input_tensor = input_tensor.to(device)

            lesion_encoded_tensor = lesion_autoencoder.get_latent(torch.clone(input_tensor))
            lesion_output_tensor = lesion_predictor(lesion_encoded_tensor)
            prl_encoded_tensor = prl_autoencoder.get_latent(torch.clone(input_tensor))
            prl_output_tensor = prl_predictor(prl_encoded_tensor)
            cvs_encoded_tensor = cvs_autoencoder.get_latent(torch.clone(input_tensor))
            cvs_output_tensor = cvs_predictor(cvs_encoded_tensor)

            lesion_loss = nn.BCELoss()(lesion_output_tensor * weight_tensor[:, 0:1], target_tensor[:, 0:1] * weight_tensor[:, 0:1])
            prl_loss = nn.BCELoss()(prl_output_tensor * weight_tensor[:, 1:2], target_tensor[:, 1:2] * weight_tensor[:, 1:2])
            cvs_loss = nn.BCELoss()(cvs_output_tensor * weight_tensor[:, 2:3], target_tensor[:, 2:3] * weight_tensor[:, 2:3])
            lesion_loss.backward()
            prl_loss.backward()
            cvs_loss.backward()
            lesion_optimizer.step()
            prl_optimizer.step()
            cvs_optimizer.step()

            batch_loss = [float(lesion_loss), float(prl_loss), float(cvs_loss)]
            epoch_loss = [epoch_loss[i] + batch_loss[i] for i in range(3)]
        print("Batch " + str(epoch_index) + ": Loss = " + str(epoch_loss))
        continue
    
    # Otherwise, train both predictor and encoder together
    for patches_batch in prl_dl.train_loader:
        lesion_joint_optimizer.zero_grad()
        prl_joint_optimizer.zero_grad()
        cvs_joint_optimizer.zero_grad()

        isolation_mask_tensor = isolate_lesion(patches_batch)
        target_tensor, weight_tensor = get_lesion_type(patches_batch, isolation_mask_tensor)
        target_tensor = target_tensor.to(device)
        weight_tensor = weight_tensor.to(device)


        input_tensor = torch.cat([patches_batch.get(key)["data"] for key in keys], dim=1) * isolation_mask_tensor
        input_tensor = input_tensor.to(device)

        lesion_encoded_tensor = lesion_autoencoder.get_latent(torch.clone(input_tensor))
        lesion_output_tensor = lesion_predictor(lesion_encoded_tensor)
        prl_encoded_tensor = prl_autoencoder.get_latent(torch.clone(input_tensor))
        prl_output_tensor = prl_predictor(prl_encoded_tensor)
        cvs_encoded_tensor = cvs_autoencoder.get_latent(torch.clone(input_tensor))
        cvs_output_tensor = cvs_predictor(cvs_encoded_tensor)

        lesion_loss = nn.BCELoss()(lesion_output_tensor * weight_tensor[:, 0:1], target_tensor[:, 0:1] * weight_tensor[:, 0:1])
        prl_loss = nn.BCELoss()(prl_output_tensor * weight_tensor[:, 1:2], target_tensor[:, 1:2] * weight_tensor[:, 1:2])
        cvs_loss = nn.BCELoss()(cvs_output_tensor * weight_tensor[:, 2:3], target_tensor[:, 2:3] * weight_tensor[:, 2:3])
        lesion_loss.backward()
        prl_loss.backward()
        cvs_loss.backward()
        lesion_joint_optimizer.step()
        prl_joint_optimizer.step()
        cvs_joint_optimizer.step()

        batch_loss = [float(lesion_loss), float(prl_loss), float(cvs_loss)]
        epoch_loss = [epoch_loss[i] + batch_loss[i] for i in range(3)]
    print("Batch " + str(epoch_index) + ": Loss = " + str(epoch_loss))
        
torch.save(lesion_autoencoder.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_individual_models/lesion_joint_05_" + str(prl_dl.i) + ".pt")
torch.save(lesion_predictor.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_individual_models/lesion_predictor_05_" + str(prl_dl.i) + ".pt")

torch.save(prl_autoencoder.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_individual_models/prl_joint_05_" + str(prl_dl.i) + ".pt")
torch.save(prl_predictor.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_individual_models/prl_predictor_05_" + str(prl_dl.i) + ".pt")

torch.save(cvs_autoencoder.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_individual_models/cvs_joint_05_" + str(prl_dl.i) + ".pt")
torch.save(cvs_predictor.state_dict(), "/home/fengling/Documents/prl/prl_pytorch/cv_individual_models/cvs_predictor_05_" + str(prl_dl.i) + ".pt")
