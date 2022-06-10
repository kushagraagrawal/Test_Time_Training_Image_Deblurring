import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F

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
    
class Encoder(nn.Module):
    def __init__(self,classifier_layer=5,res_block=ResBlock):
        super(Encoder, self ).__init__()
        
        self.conv1 = nn.Conv2d(3,4,3,padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList([res_block(3+2**i,3+2**(i+1)) for i in range(classifier_layer)])
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        for module in self.layers:
            x = module(x)
        return x,None
    
class Decoder(nn.Module):
    def __init__(self,classifier_layer=5,total_layers=8,res_block=ResBlock):
        super(Decoder, self ).__init__()
        
        self.layers = nn.ModuleList([res_block(3+2**i,3+2**(i+1)) for i in range(classifier_layer,total_layers)])
        self.final_conv = nn.Conv2d(3+2**total_layers,3,1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        
    def forward(self,x,skip_connections=None):
        for module in self.layers:
            x = module(x)
        x = self.relu(self.bn(self.final_conv(x)))
        return x
    
class Classifier(nn.Module):
    def __init__(self,classifier_layer=5,num_classes=20):
        super(Classifier, self ).__init__()
        
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3+2**classifier_layer,num_classes,3,padding=1)
        self.bn1 = nn.BatchNorm2d(num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.global_pool(x)
        return x.view(-1,self.num_classes)