""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(13, 64)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x0 = self.fc1(x)
        x0_1 = torch.transpose(x0, dim0=2, dim1=3)
        x0_2 = self.fc2(x0_1)
        x1 = self.inc(x0_2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        up1 = self.up1(x5, x4)
        # up1_shape = x
        up2 = self.up2(up1, x3)
        # up2_shape = x
        up3 = self.up3(up2, x2)
        # up3_shape = x
        up4 = self.up4(up3, x1)
        # up4_shape = x
        logits = self.outc(up4)

        # print(x0.shape)
        # print(x0_1.shape)
        # print(x0_2.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(up1_shape.shape)
        # print(up2_shape.shape)
        # print(up3_shape.shape)
        # print(up4_shape.shape)
        # print(logits.shape)
        return logits
