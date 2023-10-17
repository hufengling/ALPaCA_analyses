import torch
import torch.nn as nn

class Predictor3D(nn.Module):
    def __init__(self):
        super(Predictor3D, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(2560, 1280),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        
        self.lesion_predictor = nn.Sequential(
            nn.Linear(160, 1),
            nn.Sigmoid()
        )
        
        self.subtype_predictor = nn.Sequential(
            nn.Linear(161, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        compressed = self.predictor(x)
        lesion_pred = self.lesion_predictor(compressed)
        subtype_pred = self.subtype_predictor(torch.cat([compressed, lesion_pred], dim = 1))
        
        return torch.cat([lesion_pred, subtype_pred], dim = 1)
