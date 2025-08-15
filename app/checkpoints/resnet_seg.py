import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetSegmentation(nn.Module):
    def __init__(self, out_channels=1, pretrained=False):
        super(ResNetSegmentation, self).__init__()
        
        # ResNet18 backbone
        backbone = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final segmentation layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.up4(x)
        x = torch.relu(self.conv4(x))
        
        x = self.up3(x)
        x = torch.relu(self.conv3(x))
        
        x = self.up2(x)
        x = torch.relu(self.conv2(x))
        
        x = self.up1(x)
        x = torch.relu(self.conv1(x))
        
        # Final layer
        x = self.final(x)
        
        # Upsample to match input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)
