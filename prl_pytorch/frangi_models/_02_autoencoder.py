import torch
import torch.nn as nn

class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(5, 80, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(80), 
            nn.ReLU(), # 24x24x24x4 -> 24x24x24x32
            nn.Conv3d(80, 160, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool3d(kernel_size=2, stride=2),  
            nn.BatchNorm3d(160),
            nn.ReLU(), # 24x24x24x32 -> 12x12x12x128
            nn.Conv3d(160, 640, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(640),
            nn.ReLU(), # 12x12x12x128 -> 6x6x6x512
            nn.Conv3d(640, 2560, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(2560),
            nn.ReLU(), # 6x6x6x512 -> 3x3x3x2048
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(2560, 640, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm3d(640),
            nn.ReLU(), #3x3x3x1024 -> #6x6x6x256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(640, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(160),
            nn.ReLU(), #6x6x6x256 -> 12x12x12x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(160, 80, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm3d(80),
            nn.ReLU(), # 12x12x12x64 -> 24x24x24x16
            nn.ConvTranspose3d(80, 5, kernel_size=3, stride=1, padding=1) # 24x24x24x8 -> 24x24x24x4
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_latent(self, x):
        return self.encoder(x)
