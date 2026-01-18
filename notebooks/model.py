import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Implemented as (Conv3d -> BatchNorm3d -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """3D U-Net for liver and tumor segmentation"""
    def __init__(self, in_channels=1, out_channels=3, base_filter_count=32):
        super().__init__()
        
        # Encoder
        self.layer1 = DoubleConv(in_channels, base_filter_count)
        self.layer2 = DoubleConv(base_filter_count, base_filter_count * 2)
        self.layer3 = DoubleConv(base_filter_count * 2, base_filter_count * 4)
        self.layer4 = DoubleConv(base_filter_count * 4, base_filter_count * 8)

        self.maxpool = nn.MaxPool3d(2)
        
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.layer5 = DoubleConv((base_filter_count * 8) + (base_filter_count * 4), base_filter_count * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.layer6 = DoubleConv((base_filter_count * 4) + (base_filter_count * 2), base_filter_count * 2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.layer7 = DoubleConv((base_filter_count * 2) + base_filter_count, base_filter_count)
        
        self.layer8 = nn.Conv3d(base_filter_count, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        
        x4 = self.layer4(x3m) # Bridge
        
        x5 = self.up3(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)
        
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)
        
        x7 = self.up1(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        
        return self.layer8(x7)