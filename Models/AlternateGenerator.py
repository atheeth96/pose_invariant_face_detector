import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init

class ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding,bias,norm=nn.BatchNorm2d,activation='prelu'):
        super().__init__()
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()]
        ])
        if norm!=None:
            self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
            norm(output_channels),
            self.activations[activation])
        else:
            self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
            self.activations[activation])



    def forward(self,x):
        x = self.conv(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self,activation='prelu'):
        super().__init__()
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()],
            ['None', None]
        ])
        if self.activations[activation]:
            self.conv = nn.Sequential(
            nn.PixelShuffle(2),
            self.activations[activation])
        else:
            self.conv = nn.Sequential(
            nn.PixelShuffle(2))
        
            
    def forward(self,x):
        x = self.conv(x)
        return x
    
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,norm=nn.BatchNorm2d,activation='prelu'):
        super().__init__()
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])
        
        self.ch_out = ch_out
        self.conv = nn.Sequential(
        nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
        norm(ch_out),
        self.activations[activation],
        nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
        norm(ch_out),
        self.activations[activation]
        )

    def forward(self,x):
        
        x1=self.conv(x)
        
        return x1+x

        

class Generator(nn.Module):
    def __init__(self,n_blocks=16,downsample=True,norm=nn.BatchNorm2d):

        super().__init__()
        self.downsample=downsample
        use_bias=norm==nn.InstanceNorm2d
        self.n_blocks=n_blocks
        if downsample:
            self.conv_block_1=ConvBlock(3,64,kernel_size=4,stride=2,padding=1,bias=True,norm=None)
        else:
            self.conv_block_1=ConvBlock(3,64,kernel_size=3,stride=1,padding=1,bias=True,norm=None)

        self.res_block_list=[]
        for i in range(self.n_blocks):
            self.res_block_list.append(Recurrent_block(64))
        self.res_block=nn.Sequential(*self.res_block_list)
        self.conv_block_4=nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),\
                                        nn.BatchNorm2d(64))


        if downsample:
            self.upsample_block_1=Upsample()
            self.conv_block_5=nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1,bias=True)
        else:
            self.conv_block_5=nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,bias=True)

    
    def forward(self,input):
        x=self.conv_block_1(input)
        #64*256*256 if downsample=False; else 64*128*128


        x1=self.res_block(x)
        x1=self.conv_block_4(x1)
        #64*256*256 if downsample=False; else 64*128*128

        x=x+x1
        #64*256*256 if downsample=False; else 64*128*128

        if self.downsample:
            x=self.upsample_block_1(x)

        x=self.conv_block_5(x)
        #3*256*256
        output=nn.Tanh()(x)

        return output
    
    

    
        