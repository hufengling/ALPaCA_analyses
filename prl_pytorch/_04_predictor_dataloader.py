import os
import torch
import torchvision as tv
import torchio as tio
import random
import math
import pandas as pd

images_dir = "/home/fengling/Documents/prl/data/processed_05"
subjects_df = pd.read_csv("/home/fengling/Documents/prl/cv_output/cv_df.csv")

if os.getenv("LSB_JOBINDEX") == None or os.getenv("LSB_JOBINDEX") == "0":
    i = 1
else:
    i = int(os.getenv("LSB_JOBINDEX"))
print(i)
    
train = subjects_df["subject_id"][subjects_df["cv_index"] != i - 1].values.tolist()

# Data loader settings
transforms = [
    tio.RandomAffine(
        scales=(0.66, 1.50), 
        degrees=180,
        isotropic=True
    ),
    tio.RandomFlip(),
    tio.RandomGamma(log_gamma = (-0.5, 0.5))
    #tio.RandomElasticDeformation(num_control_points=7, locked_borders=2)
]
transform = tio.Compose(transforms)

patch_size = 24
queue_length = 250
samples_per_volume = 50
num_workers = 20

# Proposal weighting scheme
lesion_weights = {0: 1, 
                  1000: 1, 1909: 1,
                  1100: 1, 
                  1110: 1, 1119: 1,
                  1101: 1, 1109: 1,
                  1111: 1}
sampler = tio.data.LabelSampler(patch_size, "lesion_type", lesion_weights)

# Train set data loader
train_dir = [os.path.join(images_dir, x) for x in train]

train_subject_list = []
for subject_path in train_dir:
    file_paths = [os.path.join(subject_path, x) for x in os.listdir(subject_path)]
    cvs_coords_path = [s for s in file_paths if "cvs_coords" in s]
    if cvs_coords_path == []:
        contains_lesions = False
    else:
        if "cvs_coords_nl.nii.gz" in cvs_coords_path[0]:
            contains_lesions = False
        else:
            contains_lesions = True
    subject = tio.Subject(
        epi = tio.ScalarImage([s for s in file_paths if "epi_final.nii.gz" in s][0]),
        flair = tio.ScalarImage([s for s in file_paths if "flair_final.nii.gz" in s][0]),
        phase = tio.ScalarImage([s for s in file_paths if "phase_final.nii.gz" in s][0]),
        t1 = tio.ScalarImage([s for s in file_paths if "t1_final.nii.gz" in s][0]),
        prob = tio.ScalarImage([s for s in file_paths if "prob.nii.gz" in s][0]),
        lesion_mask = tio.LabelMap([s for s in file_paths if "prob_05.nii.gz" in s][0]),
        lesion_type = tio.LabelMap([s for s in file_paths if "lesion_labels.nii.gz" in s][0]),
        name = subject_path,
        contains_lesions = contains_lesions,
        contains_cvs = [False if cvs_coords_path == [] else True][0]
    )
    train_subject_list.append(subject)

train_dataset = tio.SubjectsDataset(train_subject_list, transform = transform)

train_queue = tio.Queue(
    train_dataset,
    queue_length,
    samples_per_volume,
    sampler,
    num_workers=num_workers
)

train_loader = torch.utils.data.DataLoader(
    train_queue,
    batch_size=64,
    num_workers=0,  # this must be 0
)