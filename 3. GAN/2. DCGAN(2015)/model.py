import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential( # Input: (N, channels_img, 64, 64)
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # 32 * 32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1), # 16 * 16
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1), # 8 * 8
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1), # 4 * 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0), # 1 * 1
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential( # Input: (N, z_dim, 1, 1)
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0), # (N, f_g*16, 4, 4)
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1), # (N, f_g*8, 8, 8)
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1), # (N, f_g*4, 16, 16)
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1), # (N, f_g*4, 32, 32)
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1), # (N, 3, 64, 64)
            nn.Tanh() # -1 ~ 1 pixel value
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn(N, z_dim, 1, 1)
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")

test()


