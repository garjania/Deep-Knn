import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.rep_dim = 512
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        return z


class VGG19(nn.Module):
    def __init__(self, rep_dim=512):
        super(VGG19, self).__init__()
        self.rep_dim = rep_dim
        # self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=False)
        self.lin = torch.Linear(1000, self.rep_dim)

    def forward(self, x):
        # z = self.conv(x)
        z = self.vgg(x)
        z = self.lin(z)
        return z
