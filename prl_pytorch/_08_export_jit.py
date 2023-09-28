#bsub -q taki_normal -J "make_predictions[1-970]" -o ~/Documents/prl/stdout/make_predictions.txt bash -c "python /home/fengling/Documents/prl/prl_pytorch/_07_make_predictions_individual.py"
#bsub -q lpcgpu -gpu "num=1" -n 1 -J "make_predictions[1-970]" -o ~/Documents/prl/stdout/make_predictions_nograd.txt bash -c "python /home/fengling/Documents/prl/prl_pytorch/_07_make_predictions_individual.py"
import os
import torch
import torch.nn as nn
import torchvision as tv
import torchio as tio
import numpy as np
import random
import pandas as pd

import _01_dataloader_individual as prl_dl
import _02_autoencoder as prl_ae
import _05_predictor as prl_pred

#importlib.reload(prl_ae)
#importlib.reload(prl_pred)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

for i in range(1, 11):
    print(i)
    prl_autoencoder = prl_ae.Autoencoder3D()
    prl_autoencoder_joint = prl_ae.Autoencoder3D()
    prl_predictor = prl_pred.Predictor3D()

    keys = ["t1", "flair", "epi", "phase"]

    # Load the state_dict from the checkpoint
    prl_autoencoder.load_state_dict(torch.load("/home/fengling/Documents/prl/prl_pytorch/cv_models/prl_autoencoder_05_" + str(prl_dl.cv_index) + ".pt", 
                                               map_location=torch.device('cpu')))
    prl_autoencoder.eval()
    prl_autoencoder.to(device)

    # Load the state_dict from the checkpoint
    prl_autoencoder_joint.load_state_dict(torch.load("/home/fengling/Documents/prl/prl_pytorch/cv_models/prl_joint_05_" + str(prl_dl.cv_index) + ".pt", 
                                               map_location=torch.device('cpu')))
    prl_autoencoder_joint.eval()
    prl_autoencoder_joint.to(device)

    # Load the state_dict from the checkpoint
    prl_predictor.load_state_dict(torch.load("/home/fengling/Documents/prl/prl_pytorch/cv_models/prl_predictor_05_" + str(prl_dl.cv_index) + ".pt", 
                                               map_location=torch.device('cpu')))
    prl_predictor.eval()
    prl_predictor.to(device)
    
    autoencoder_trace = torch.jit.trace(prl_autoencoder_joint, torch.randn([1, 4, 24, 24, 24]).detach())
    predictor_trace = torch.jit.trace(prl_predictor, torch.randn([1, 1024, 3, 3, 3]).detach())
    
    torch.jit.save(autoencoder_trace, "./trace_models/autoencoder_" + str(i) + ".pt")
    torch.jit.save(predictor_trace, "./trace_models/predictor_" + str(i) + ".pt")