import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,channels_img,features_d):
        super(Discriminator, self).__init__()
        self.disc=nn.Sequential(
            nn.Conv2d(channels_img,features_d,kernel_size=4, stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False),
                      #No Bias is needed as we are performing batch_norm
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,z_dim,channels_img,features_g):
        super(Generator,self).__init__()
        self.gen=nn.Sequential(
            #Input: N x z_dim x 1 x 1  N is batch size
            self._block(z_dim,features_g*16,4,1,0), #N xf_g x4 x4
            self._block(features_g*16,features_g*8,4,2,1),
            self._block(features_g*8,features_g*4,4,2,1),
            self._block(features_g*4,features_g*2,4,2,1),
            nn.ConvTranspose2d(features_g*2,channels_img,kernel_size=4,stride=2,padding=1),
            nn.Tanh(),
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False),
                #No Bias is needed as we are performing batch_norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self,z):
        return self.gen(z)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N , in_channels, H, W=8,3,64,64
    z_dim=100
    x=torch.randn((N,in_channels,H,W))
    z=torch.randn((N,z_dim,1,1))
    disc=Discriminator(in_channels,8)
    gen=Generator(z_dim,in_channels,64)
    initialize_weights(disc)
    print(disc(x).shape)
    print(gen(z).shape)
    assert disc(x).shape ==(N,1,1,1)

test()
