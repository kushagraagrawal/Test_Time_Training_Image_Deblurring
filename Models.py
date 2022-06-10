import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F

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
    
    
    
class ResBlock(nn.Module ):
    def __init__(self, inch, outch, stride=1, dilation=1 ):
        # Residual Block
        # inch: input feature channel
        # outch: output feature channel
        # stride: the stride of  convolution layer
        super(ResBlock, self ).__init__()
        assert(stride == 1 or stride == 2 )

        self.conv1 = nn.Conv2d(inch, outch, 3, stride, padding = dilation, bias=False,
                dilation = dilation )
        self.bn1 = nn.BatchNorm2d(outch )
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, padding = dilation, bias=False,
                dilation = dilation )
        self.bn2 = nn.BatchNorm2d(outch )

        if inch != outch:
            self.mapping = nn.Sequential(
                        nn.Conv2d(inch, outch, 1, stride, bias=False ),
                        nn.BatchNorm2d(outch )
                    )
        else:
            self.mapping = None

    def forward(self, x ):
        y = x
        if not self.mapping is None:
            y = self.mapping(y )

        out = F.relu(self.bn1(self.conv1(x) ), inplace=True )
        out = self.bn2(self.conv2(out ) )

        out += y
        out = F.relu(out, inplace=True )

        return out
    
class encoderSPP(nn.Module ):
    def __init__(self ):
        super(encoderSPP, self ).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, dilation=2)
        self.b3_2 = ResBlock(256, 256, 1, dilation=2)

        self.b4_1 = ResBlock(256, 512, 1, dilation=4)
        self.b4_2 = ResBlock(512, 512, 1, dilation=4)
        
        self.pyramid_pool1 = nn.AdaptiveAvgPool2d((1,1))
        self.pyramid_conv1 = nn.Conv2d(512,128,1)
        self.pyramid_bn1 = nn.BatchNorm2d(128)
        
        self.pyramid_pool2 = nn.AdaptiveAvgPool2d((2,2))
        self.pyramid_conv2 = nn.Conv2d(512,128,1)
        self.pyramid_bn2 = nn.BatchNorm2d(128)
        
        self.pyramid_pool3 = nn.AdaptiveAvgPool2d((3,3))
        self.pyramid_conv3 = nn.Conv2d(512,128,1)
        self.pyramid_bn3 = nn.BatchNorm2d(128)
        
        self.pyramid_pool4 = nn.AdaptiveAvgPool2d((6,6))
        self.pyramid_conv4 = nn.Conv2d(512,128,1)
        self.pyramid_bn4 = nn.BatchNorm2d(128)
        
    def forward(self, im ):

        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.b4_2(self.b4_1(x4 ) )
        
        p1 = self.pyramid_bn1(self.pyramid_conv1(self.pyramid_pool1(x5)))
        p2 = self.pyramid_bn2(self.pyramid_conv2(self.pyramid_pool2(x5)))
        p3 = self.pyramid_bn3(self.pyramid_conv3(self.pyramid_pool3(x5)))
        p4 = self.pyramid_bn4(self.pyramid_conv4(self.pyramid_pool4(x5)))
        
        _, _, nh, nw = x5.size()
        p1 = F.interpolate(p1,[nh, nw], mode='bilinear',align_corners=True)
        p2 = F.interpolate(p2,[nh, nw], mode='bilinear',align_corners=True)
        p3 = F.interpolate(p3,[nh, nw], mode='bilinear',align_corners=True)
        p4 = F.interpolate(p4,[nh, nw], mode='bilinear',align_corners=True)
        
        x5 = torch.cat([p1,p2,p3,p4,x5],dim=1)

        skip_connections = [x1, x2, x3, x4, x5]
        return x5,skip_connections
    
class decoderSPP(nn.Module ):
    def __init__(self, isSpp = False ):
        super(decoderSPP, self).__init__()

        self.conv1 = nn.Conv2d(1024+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 3, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(64+3, 3, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(3 )
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(3 )
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1, bias=False )

    def forward(self, im, skip_connections):
        
        x1, x2, x3, x4, x5 = skip_connections

        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear' ,align_corners=True)
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear',align_corners=True )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x5], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear',align_corners=True)
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' ,align_corners=True)
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.conv4(y3)

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear',align_corners=True)

        return y4
    
def loadPretrainedWeight(network, isOutput = False ):
    paramList = []
    resnet18 = resnet.resnet18(pretrained=True )
    for param in resnet18.parameters():
        paramList.append(param )

    cnt = 0
    for param in network.parameters():
        if paramList[cnt ].size() == param.size():
            param.data.copy_(paramList[cnt].data )
            #param.data.copy_(param.data )
            if isOutput:
                print(param.size() )
        else:
            print(param.shape, paramList[cnt].shape )
            break
        cnt += 1
    
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
        self.fc = nn.Linear(1000,20) # 21 classes
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x