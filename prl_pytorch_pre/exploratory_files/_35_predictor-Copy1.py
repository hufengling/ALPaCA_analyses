import torch
import torch.nn as nn

class Predictor3D(nn.Module):
    def __init__(self):
        super(Predictor3D, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predictor(x)