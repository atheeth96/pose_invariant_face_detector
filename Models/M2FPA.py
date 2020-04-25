import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init


def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def save_model(model,optimizer,name,scheduler=None):
    if scheduler==None:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    else:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()}

    torch.save(checkpoint,name)

def load_model(filename,model,optimizer=None,scheduler=None):
    checkpoint=torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("Done loading")
    if  optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
    if  scheduler:
        scheduler.load_state_dict(checkpoint['optimizer'])
        print(scheduler.state_dict()['param_groups'][-1]['lr'],' : Learning rate')


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method {} is not implemented in pytorch'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialized network with {} initialization'.format(init_type))
    net.apply(init_func)
    
    
class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m
    
    
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3,activation='relu',stride=1,padding=1,bias=True,norm='batch'):
        super().__init__()
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)]
        ])
        self.norm = nn.ModuleDict([
                ['batch', nn.BatchNorm2d(ch_out)],
                ['instance', nn.GroupNorm(ch_out, ch_out)],
            ['layer', nn.GroupNorm(1, ch_out)]
        ])
        
        
        self.conv = nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
        self.norm[norm],
        self.activations[activation]
       )


    def forward(self,x):
        x = self.conv(x)
        return x
    
    
class Recurrent_block(nn.Module):
    def __init__(self,ch_in):
        super().__init__()
        
        self.ch_out = ch_in
        self.conv1 = conv_block(ch_in,ch_in,kernel_size=3,activation='relu',stride=1,padding=1,bias=True,norm='batch')
        self.conv2 = conv_block(ch_in,ch_in,kernel_size=3,activation='relu',stride=1,padding=1,bias=True,norm='batch')

    def forward(self,x):
        
        x1=self.conv1(x)
        x1=self.conv2(x1)
        
        return x+x1

    
class encoder_block(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3,activation='relu',stride=1,padding=1,bias=True,norm='batch'):
        super().__init__()
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        
        self.conv_block = conv_block(ch_in,ch_out,kernel_size=kernel_size\
                                     ,activation=activation,stride=stride,padding=padding,bias=bias,norm=norm)
        self.residual_block = Recurrent_block(ch_out)

    def forward(self,x):
        
        x=self.conv_block(x)
        x=self.residual_block(x)
        
        return x

class up_conv(nn.Module):
    def __init__(self,n_channels, out_channels, kernel_size, stride=1,norm='batch',activation='relu'):
        super().__init__()
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)]
        ])
        self.norm = nn.ModuleDict([
                ['batch', nn.BatchNorm2d(out_channels)],
                ['instance', nn.GroupNorm(out_channels, out_channels)],
            ['layer', nn.GroupNorm(1, out_channels)]
        ])
        self.up = nn.Sequential(
        nn.ConvTranspose2d(n_channels, out_channels, kernel_size, stride=stride),
        self.norm[norm],
        self.activations[activation]
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
    
class Generator(nn.Module):
    def __init__(self,norm='batch',activation='relu'):
        super().__init__()
        
        self.encoder_block_0=encoder_block(3,64,kernel_size=7,\
                                           activation=activation,stride=1,padding=3,bias=True,norm=norm)
        self.encoder_block_1=encoder_block(64,64,kernel_size=5,\
                                           activation=activation,stride=2,padding=2,bias=True,norm=norm)
        self.encoder_block_2=encoder_block(64,128,kernel_size=3,\
                                           activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.encoder_block_3=encoder_block(128,256,kernel_size=3,\
                                           activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.encoder_block_4=encoder_block(256,512,kernel_size=3,\
                                           activation=activation,stride=2,padding=1,bias=True,norm=norm)
        
        self.fc1=nn.Linear(131072, 512, bias=True)
        self.maxout=Maxout(512,128,3)
        self.fc2=nn.Linear(128, 16384, bias=True)
        
        self.dec0_1=up_conv(64, 32, 4, stride=4,norm=norm,activation=activation)
        self.dec0_2=up_conv(32, 16, 2, stride=2,norm=norm,activation=activation)
        self.dec0_3=up_conv(16, 8, 2, stride=2,norm=norm,activation=activation)
        
        self.dec1=nn.Sequential(up_conv(576, 512, 2, stride=2,norm=norm,activation=activation),\
                                Recurrent_block(512))
                                
        self.dec2=nn.Sequential(up_conv(768, 256, 2, stride=2,norm=norm,activation=activation),\
                                Recurrent_block(256))
        self.dec3=nn.Sequential(up_conv(416, 128, 2, stride=2,norm=norm,activation=activation),\
                                Recurrent_block(128))
        self.dec4=nn.Sequential(up_conv(208, 64, 2, stride=2,norm=norm,activation=activation),\
                                Recurrent_block(64))
        
#         self.conv5=conv_block(256,64,kernel_size=3,activation=activation,stride=1,padding=1,bias=True,norm=norm)
#         self.conv6=conv_block(128,32,kernel_size=3,activation=activation,stride=1,padding=1,bias=True,norm=norm)
        self.conv7=conv_block(139,3,kernel_size=5,activation=activation,stride=1,padding=2,bias=True,norm=norm)
        self.conv8=conv_block(3,3,kernel_size=3,activation=activation,stride=1,padding=1,bias=True,norm=norm)
        
        self.conv9=nn.Conv2d(3, 3, kernel_size=3,stride=1,padding=1,bias=True)
 
        self.final_act=nn.Tanh()

    def forward(self,x):
        batch_size=x.size()[0]
        x0 = self.encoder_block_0(x)
        x1 = self.encoder_block_1(x0)
        x2 = self.encoder_block_2(x1)
        x3 = self.encoder_block_3(x2)
        
        x4 = self.encoder_block_4(x3)
        
        x_temp = self.fc1(x4.view((batch_size,-1)))
        x_temp = self.maxout(x_temp)
        x_temp = self.fc2(x_temp)
        x_fc2  = x_temp.view((batch_size,64,16,16))
        
        x_dec0_1 = self.dec0_1(x_fc2)
        x_dec0_2 = self.dec0_2(x_dec0_1)
        x_dec0_3 = self.dec0_3(x_dec0_2)
        x_temp = self.dec1(torch.cat((x_fc2,x4),dim=1))
        x_temp = self.dec2(torch.cat((x_temp,x3),dim=1))
        
        x_dec3 = self.dec3(torch.cat((x_temp,x2,x_dec0_1),dim=1))
        x_dec4 = self.dec4(torch.cat((x_dec3,x1,x_dec0_2),dim=1))
        
#         x_temp = self.conv5(x_dec4)
#         x_temp = self.conv6(x_temp)
        
        x_temp = self.conv7(torch.cat((x_dec4,x0,x,x_dec0_3),dim=1))
        x_temp = self.conv8(x_temp)
        x = self.conv9(x_temp)
        x=self.final_act(x)
  
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self,norm='batch',activation='relu'):
        super().__init__()
        self.conv1=conv_block(ch_in=3,ch_out=64,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.conv2=conv_block(ch_in=64,ch_out=128,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.conv3=conv_block(ch_in=128,ch_out=256,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.conv4=conv_block(ch_in=256,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.conv5=conv_block(ch_in=512,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.conv6=conv_block(ch_in=512,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm)
        self.conv7=nn.Conv2d(512,1, kernel_size=1,stride=1,padding=0,bias=True)
        
    def forward(self,x):
        
        x=self.conv1(x)

        x=self.conv2(x)
        
        x=self.conv3(x)
        
        x=self.conv4(x)
        
        x=self.conv5(x)
        
        x=self.conv6(x)
        
        x=self.conv7(x)
        
        x=torch.sigmoid(x)
        
        return x
    
class ParserDiscriminator(nn.Module):
    def __init__(self,norm='batch',activation='relu'):
        super().__init__()
        
        self.feat_extractor_1=nn.Sequential(\
                                            conv_block(ch_in=3,ch_out=64,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=64,ch_out=128,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=128,ch_out=256,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=256,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=512,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm))
        
        self.feat_extractor_2=nn.Sequential(\
                                            conv_block(ch_in=3,ch_out=64,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=64,ch_out=128,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=128,ch_out=256,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=256,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=512,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm))
        
        self.feat_extractor_3=nn.Sequential(\
                                            conv_block(ch_in=3,ch_out=64,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=64,ch_out=128,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=128,ch_out=256,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=256,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=512,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm))
        
        self.fc=nn.Sequential(\
                                            conv_block(ch_in=512*3,ch_out=512,kernel_size=3,activation=activation,stride=1,padding=1,bias=True,norm=norm),\
                                            conv_block(ch_in=512,ch_out=512,kernel_size=3,activation=activation,stride=2,padding=1,bias=True,norm=norm),\
                                            nn.Conv2d(512, 1, kernel_size=3,stride=1,padding=1,bias=True))
        
    def forward(self,x,y,z):
        
        f1=self.feat_extractor_1(x)
        f2=self.feat_extractor_1(y)
        f3=self.feat_extractor_1(z)
        op=self.fc(torch.cat((f1,f2,f3),dim=1))
        output=torch.sigmoid(op)

        return op

        
    