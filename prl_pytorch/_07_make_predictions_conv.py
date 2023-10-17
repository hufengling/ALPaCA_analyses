#bsub -q taki_normal -J "make_predictions[1-970]" -o ~/Documents/prl/stdout/make_predictions_no_frangi.txt bash -c "python /home/fengling/Documents/prl/prl_pytorch/_07_make_predictions_individual_no_frangi.py"
#bsub -q lpcgpu -gpu "num=1" -n 1 -J "make_predictions[1-970]" -o ~/Documents/prl/stdout/make_predictions_conv.txt bash -c "python /home/fengling/Documents/prl/prl_pytorch/_07_make_predictions_conv.py"
import os
import torch
import torch.nn as nn
import torchvision as tv
import torchio as tio
import numpy as np
import random
import pandas as pd
import gc

import _01_dataloader_individual_no_frangi as prl_dl
import _02_autoencoder_conv as prl_ae
import _05_predictor_conv as prl_pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

prl_autoencoder_joint = prl_ae.Autoencoder3D()
prl_predictor = prl_pred.Predictor3D()

keys = ["t1", "flair", "epi", "phase"]

# Load the state_dict from the checkpoint
prl_autoencoder_joint.load_state_dict(torch.load("/home/fengling/Documents/prl/prl_pytorch/cv_models_conv/prl_joint_05_" + str(prl_dl.cv_index) + ".pt", 
                                           map_location=torch.device('cpu')))
prl_autoencoder_joint.eval()
prl_autoencoder_joint.to(device)

# Load the state_dict from the checkpoint
prl_predictor.load_state_dict(torch.load("/home/fengling/Documents/prl/prl_pytorch/cv_models_conv/prl_predictor_05_" + str(prl_dl.cv_index) + ".pt", 
                                           map_location=torch.device('cpu')))
prl_predictor.eval()
prl_predictor.to(device)

def process_lesion_type(lesion_id, contains_lesions, contains_cvs): # Return tensor of [is_lesion, is_PRL, is_CVS]
    target = torch.zeros(3)
    weight = torch.ones(3)
    
    if contains_lesions == False:
        weight[0] = 0
    if contains_cvs == False:
        weight[2] = 0
   
    digits = [int(x) for x in str(lesion_id)] 
    # First digit is always 1 for computational convenience
   
    if (lesion_id == 0):
        return([target, weight]) # Just return the tensor of 0s
    
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

def get_coords(candidate_id, num_coords, 
               lesion_mask):
    candidate_coords = torch.nonzero(lesion_mask == candidate_id)
    max_coords = min(num_coords, candidate_coords.size()[0])
    random_inds = random.sample(range(candidate_coords.size()[0]), max_coords)
    return(candidate_coords[random_inds, :])

def isolate_lesion(lesion_mask_patch, candidate_id):
    isolation_mask = 0.1 + 0.9 * ((lesion_mask_patch == 0) + (lesion_mask_patch == candidate_id))

    if (isolation_mask == 0.1).any():
        return([True, isolation_mask])
    else:
        return([False, isolation_mask])
    
def rotate_patch(patch, invert, face, rotations):
    if (invert == 1):
        patch = torch.flip(patch, 1) # Reflect

    if (face == 2):
        patch = torch.transpose(torch.flip(patch, 1), 1, 2) # Flip cube towards
    
    if (face == 3):
        patch = torch.transpose(torch.flip(patch, 1), 1, 2) # Flip cube towards twice
        patch = torch.transpose(torch.flip(patch, 1), 1, 2)

    if (face == 4):
        patch = torch.transpose(torch.flip(patch, 2), 2, 1) # Flip cube away

    if (face == 5):
        patch = torch.transpose(torch.flip(patch, 3), 1, 3) # Flip cube to left

    if (face == 6):
        patch = torch.transpose(torch.flip(patch, 1), 3, 1) # Flip cube to right

    if (rotations == 1):
        patch = torch.transpose(torch.flip(patch, 2), 2, 3)

    if (rotations == 2):
        patch = torch.transpose(torch.flip(patch, 2), 2, 3)
        patch = torch.transpose(torch.flip(patch, 2), 2, 3)

    if (rotations == 3):
        patch = torch.transpose(torch.flip(patch, 3), 3, 2)

    return(patch)

def extract_patch(coord, candidate_id, 
                  t1, flair, epi, phase,
                  lesion_mask, lesion_type, rotate_patches = True):
    start_ends = [coord[1] - 12, coord[1] + 12, 
                  coord[2] - 12, coord[2] + 12,
                  coord[3] - 12, coord[3] + 12]
    x_start = max(start_ends[0], 0)
    x_end = min(start_ends[1], t1.size()[1] - 1)
    y_start = max(start_ends[2], 0)
    y_end = min(start_ends[3], t1.size()[2] - 1)
    z_start = max(start_ends[4], 0)
    z_end = min(start_ends[5], t1.size()[3] - 1)

    t1_patch = t1[:, x_start:x_end, y_start:y_end, z_start:z_end]
    flair_patch = flair[:, x_start:x_end, y_start:y_end, z_start:z_end]
    epi_patch = epi[:, x_start:x_end, y_start:y_end, z_start:z_end]
    phase_patch = phase[:, x_start:x_end, y_start:y_end, z_start:z_end]

    patch = torch.cat((t1_patch, flair_patch, epi_patch, phase_patch))
    lesion_mask_patch = lesion_mask[:, x_start:x_end, y_start:y_end, z_start:z_end]
    lesion_type_patch = lesion_type[:, x_start:x_end, y_start:y_end, z_start:z_end]
    if tuple(patch.size()) != (len(keys), 24, 24, 24):
        patch, lesion_mask_patch, lesion_type_patch = pad_patches(patch, lesion_mask_patch, 
                                                                  lesion_type_patch, start_ends, t1)
    is_multiple, isolation_mask = isolate_lesion(lesion_mask_patch, candidate_id)

    if rotate_patches:
        invert = random.sample(range(0, 2), 1) # Mirror patch
        face = random.sample(range(1, 7), 1) # Which face of the tensor is "down"
        rotations = random.sample(range(0, 4), 1) # Rotate tensor radially once correct face is "down"
    
        patch = torch.cat((rotate_patch(patch[0, :, :, :], invert, face, rotations).unsqueeze_(0),
                           rotate_patch(patch[1, :, :, :], invert, face, rotations).unsqueeze_(0),
                           rotate_patch(patch[2, :, :, :], invert, face, rotations).unsqueeze_(0),
                           rotate_patch(patch[3, :, :, :], invert, face, rotations).unsqueeze_(0)))
        isolation_mask = rotate_patch(isolation_mask, invert, face, rotations)
        
    if is_multiple:
        patch = patch * (isolation_mask.repeat(len(keys), 1, 1, 1))
        lesion_id = (lesion_type_patch * isolation_mask).unique()
        lesion_id = lesion_id[lesion_id != 0]
    else:
        lesion_id = lesion_type_patch.unique()
        lesion_id = lesion_id[lesion_id != 0]

    return([patch, lesion_id])

def pad_patches(patch, lesion_mask_patch, lesion_type_patch, 
                start_ends, t1):
    patch_pad_tensor = torch.zeros(len(keys), 24, 24, 24)
    mask_pad_tensor = torch.zeros(1, 24, 24, 24)
    type_pad_tensor = torch.zeros(1, 24, 24, 24)
    starts = [start_ends[i] for i in [0, 2, 4]]
    start_patch = [0 - start if start < 0 else 0 for start in starts]
    ends = [start_ends[i] for i in [1, 3, 5]]
    end_patch = [23 - (ends[i] - t1.size()[i + 1]) if ends[i] >= t1.size()[i + 1] else 24 for i in range(len(ends))]

    patch_pad_tensor[:,
                     start_patch[0]:end_patch[0], 
                     start_patch[1]:end_patch[1], 
                     start_patch[2]:end_patch[2]] = patch
    mask_pad_tensor[:,
                     start_patch[0]:end_patch[0], 
                     start_patch[1]:end_patch[1], 
                     start_patch[2]:end_patch[2]] = lesion_mask_patch
    type_pad_tensor[:,
                     start_patch[0]:end_patch[0], 
                     start_patch[1]:end_patch[1], 
                     start_patch[2]:end_patch[2]] = lesion_type_patch
    
    return(patch_pad_tensor, mask_pad_tensor, type_pad_tensor)

def get_predictions(subject, num_coords):
    name = subject["name"]
    print(name)

    lesion_mask = subject["lesion_mask"]["data"]
    lesion_type = subject["lesion_type"]["data"]

    t1 = subject["t1"]["data"]
    flair = subject["flair"]["data"]
    epi = subject["epi"]["data"]
    phase = subject["phase"]["data"]
    
    contains_lesions = subject["contains_lesions"]
    contains_cvs = subject["contains_cvs"]

    output_tensor = torch.zeros(int(lesion_mask.max()), 3)
    variance_tensor = torch.zeros(int(lesion_mask.max()), 3)
    target_tensor = torch.zeros(int(lesion_mask.max()), 3)
    weight_tensor = torch.zeros(int(lesion_mask.max()), 3)
    print("Making predictions for " + str(lesion_mask.max()) + " lesions")
    
    for candidate_id in range(1, int(lesion_mask.max()) + 1):
        print("Lesion " + str(candidate_id), flush=True)
        coords = get_coords(candidate_id, num_coords, lesion_mask)
        tmp_coords = coords[0, :]
        target_tensor[candidate_id - 1, :], weight_tensor[candidate_id - 1, :] = process_lesion_type(int(lesion_type[tmp_coords[0],
                                                                                                                     tmp_coords[1],
                                                                                                                     tmp_coords[2],
                                                                                                                     tmp_coords[3]]),
                                                                                                     contains_lesions,
                                                                                                     contains_cvs)

        all_patch = torch.zeros(coords.size()[0], len(keys), 24, 24, 24)
        for i in range(coords.size()[0]):
            coord = coords[i, :]
            patch, lesion_id = extract_patch(coord, candidate_id, 
                                             t1, flair, epi, phase,
                                            lesion_mask, lesion_type)
            all_patch[i, :, :, :, :] = patch
            
        all_patch = all_patch.to(device)
        with torch.no_grad():
            output = prl_predictor(prl_autoencoder_joint.encoder(all_patch)) 
        prediction = output.to("cpu")

        if num_coords > 1:
            output_tensor[candidate_id - 1, :] = torch.mean(prediction, dim=0)
            variance_tensor[candidate_id - 1, :] = torch.var(prediction, dim=0)
        else:
            output_tensor[candidate_id - 1, :] = prediction
            variance_tensor[candidate_id - 1, :] = torch.zeros(1, 3)

    output_df = pd.DataFrame(torch.cat([output_tensor, 
                                        variance_tensor, 
                                        target_tensor, 
                                        weight_tensor], dim=1).detach().numpy())
    output_df["subject"] = name
    return(output_df)


df = get_predictions(prl_dl.subject, 50)

if not os.path.exists("/home/fengling/Documents/prl/cv_output_conv/split_" + str(prl_dl.cv_index)):
    os.mkdir("/home/fengling/Documents/prl/cv_output_conv/split_" + str(prl_dl.cv_index))
    
df.to_csv("/home/fengling/Documents/prl/cv_output_conv/split_" + str(prl_dl.cv_index) + 
          "/subject_" + str(prl_dl.subject_id) + ".csv")
