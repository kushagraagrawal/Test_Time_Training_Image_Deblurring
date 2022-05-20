import torch
import torch.nn as nn

class SameConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SameConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x
    
class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSampleConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,3,padding=1,stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x
    
    
class UpSampleConv(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(UpSampleConv, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,3,padding=1,stride=2,output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.bn(self.deconv(x)))
        return x
    
class Unet_encoder(nn.Module):

    def __init__(self, in_channels):
        super(Unet_encoder, self).__init__()

        #encoder downsample convs
        self.encoders_downsample = nn.ModuleList([
            DownSampleConv(64, 128),  # bs x 128 x H/2 x W/2
            DownSampleConv(128, 256),  # bs x 256 x H/4 x W/4
            DownSampleConv(256, 512),  # bs x 512 x H/8 x W/8
            DownSampleConv(512, 1024),  # bs x 512 x H/16 x W/16
        ])
        
        #encoder same convs
        self.encoders_same = nn.ModuleList([
            SameConv(in_channels, 64),  # bs x 64 x H x W
            SameConv(128, 128),  # bs x 128 x H/2 x W/2
            SameConv(256, 256),  # bs x 256 x H/4 x W/4
            SameConv(512, 512),  # bs x 512 x H/8 x W/8
        ])

    def forward(self, x):
        skips_cons = []
        for encoder_down,encoder_same in zip(self.encoders_downsample,self.encoders_same):
            x = encoder_same(x)
            skips_cons.append(x)
            x = encoder_down(x)

        skips_cons = list(reversed(skips_cons))
        return x,skips_cons
    
class Unet_decoder(nn.Module):

    def __init__(self,out_channels=3):
        super(Unet_decoder, self).__init__()

        #decoder downsample convs
        self.decoders_upsample = nn.ModuleList([
            UpSampleConv(1024, 512),  # bs x 1024 x H/8 x W/8
            UpSampleConv(512, 256),  # bs x 256 x H/4 x W/4
            UpSampleConv(256, 128),  # bs x 128 x H/2 x W/2
            UpSampleConv(128, 64),  # bs x 64 x H x W
        ])
        
        #decoder same convs
        self.decoder_same = nn.ModuleList([
            SameConv(1024,512),  # bs x 512 x H/8 x W/8
            SameConv(512,256),  # bs x 256 x H/4 x W/4
            SameConv(256,128),  # bs x 128 x H/2 x W/2
            SameConv(128,64),  # bs x 64 x H x W
        ])

        self.final_conv = nn.Conv2d(64,out_channels,3,padding=1)
        
    def forward(self,x,skip_cons):
        for decoder_up,decoder_same,skip in zip(self.decoders_upsample,self.decoder_same,skip_cons):
            x = decoder_up(x)
            x = torch.cat((x,skip),axis=1)
            x = decoder_same(x)
        x = self.final_conv(x)
        return x
    
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1024,2048,3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(2048)
        
        self.conv2 = nn.Conv2d(2048,4096,3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(4096)
        
        self.conv3 = nn.Conv2d(4096,1000,1)
        self.bn3 = nn.BatchNorm2d(1000)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1000,21) # 21 classes
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x