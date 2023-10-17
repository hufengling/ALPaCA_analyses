import torch
import torch.nn as nn

class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(16), 
            nn.ReLU(), # 24x24x24x4 -> 24x24x24x16
            nn.Conv3d(16, 32, kernel_size=5, stride = 1, padding=2),
            nn.BatchNorm3d(32), 
            nn.ReLU(), # 24x24x24x16 -> 24x24x24x32
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool3d(kernel_size=2, stride=2),  
            nn.BatchNorm3d(64),
            nn.ReLU(), # 24x24x24x32 -> 12x12x12x64
            nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(), # 12x12x12x64 -> 6x6x6x256
            nn.Conv3d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(1024),
            nn.ReLU(), # 6x6x6x256 -> 3x3x3x1024
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(1024, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(256),
            nn.ReLU(), #3x3x3x1024 -> #6x6x6x256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(), #6x6x6x256 -> 12x12x12x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(64, 16, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm3d(16),
            nn.ReLU(), # 12x12x12x64 -> 24x24x24x16
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), # 24x24x24x16 -> 24x24x24x8
            nn.ConvTranspose3d(8, 4, kernel_size=3, stride=1, padding=1) # 24x24x24x8 -> 24x24x24x4
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_latent(self, x):
        return self.encoder(x)
