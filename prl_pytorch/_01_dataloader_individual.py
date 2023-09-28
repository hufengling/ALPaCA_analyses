import os
import torch
import torchvision as tv
import torchio as tio
import random
import pandas as pd

images_dir = "/home/fengling/Documents/prl/data/processed_05"
subjects_df = pd.read_csv("/home/fengling/Documents/prl/cv_output/cv_df.csv")

if os.getenv("LSB_JOBINDEX") == None or os.getenv("LSB_JOBINDEX") == "0":
    i = 4
else:
    i = int(os.getenv("LSB_JOBINDEX"))
    
cv_index = i % (max(subjects_df["cv_index"].tolist()) + 1) + 1
subject_id = (i - 1) % subjects_df.shape[0]
train = subjects_df["subject_id"][subject_id]

print(train)
print(cv_index)
print(subject_id)

output_path = "/home/fengling/Documents/prl/cv_output/split_" + str(cv_index)

if not os.path.exists(output_path):
    os.makedirs(output_path)

patch_size = 24

# Load individual
subject_path = os.path.join(images_dir, train)

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
