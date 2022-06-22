from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3, ngf=64, n_classes=13):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.ngf = ngf

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels=None):
        noise = noise.view(noise.shape[0], noise.shape[1], 1, 1)

        return self.generator(noise)


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, ndf=64, n_classes=13):
        super(Discriminator, self).__init__()

        self.num_channels = num_channels
        self.ndf = ndf

        self.main = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(self.num_channels, self.ndf, 3, 1, 1, bias=False)),
                ("lrelu1", nn.LeakyReLU(0.2, inplace=True)),

                ("conv2", nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
                ("batchnorm2", nn.BatchNorm2d(self.ndf * 2)),
                ("lrelu2", nn.LeakyReLU(0.2, inplace=True)),

                ("conv3", nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
                ("batchnorm3", nn.BatchNorm2d(self.ndf * 4)),
                ("lrelu3", nn.LeakyReLU(0.2, inplace=True)),

                ("conv4", nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
                ("batchnorm4", nn.BatchNorm2d(self.ndf * 8)),
                ("lrelu4", nn.LeakyReLU(0.2, inplace=True)),
            ])
        )

        self.fc = nn.Linear((ndf * 8) * 8 * 8, 1)

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, (self.ndf * 8) * 8 * 8)
        output = torch.sigmoid(self.fc(x))
        return output.squeeze()


class DiscriminatorDropout(nn.Module):
    def __init__(self, num_channels=3, ndf=64, n_classes=13):
        super(DiscriminatorDropout, self).__init__()

        self.num_channels = num_channels
        self.ndf = ndf

        self.main = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(self.num_channels, self.ndf, 3, 1, 1, bias=False)),
                ("lrelu1", nn.LeakyReLU(0.2, inplace=True)),
                ("dropout1", nn.Dropout(0.2)),

                ("conv2", nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
                ("batchnorm2", nn.BatchNorm2d(self.ndf * 2)),
                ("lrelu2", nn.LeakyReLU(0.2, inplace=True)),
                ("dropout2", nn.Dropout(0.2)),

                ("conv3", nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
                ("batchnorm3", nn.BatchNorm2d(self.ndf * 4)),
                ("lrelu3", nn.LeakyReLU(0.2, inplace=True)),
                ("dropout3", nn.Dropout(0.2)),

                ("conv4", nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
                ("batchnorm4", nn.BatchNorm2d(self.ndf * 8)),
                ("lrelu4", nn.LeakyReLU(0.2, inplace=True)),
                ("dropout4", nn.Dropout(0.2)),
            ])
        )
        self.fc = nn.Linear((ndf * 8) * 8 * 8, 1)

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, (self.ndf * 8) * 8 * 8)
        output = torch.sigmoid(self.fc(x))
        return output.squeeze()


class DiscriminatorSN(nn.Module):
    def __init__(self, num_channels=3, ndf=64, n_classes=13):
        super(DiscriminatorSN, self).__init__()

        self.num_channels = num_channels
        self.ndf = ndf

        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", spectral_norm(nn.Conv2d(num_channels, ndf, 3, 1, 1, bias=False))),
                    ("lrelu1", nn.LeakyReLU(0.2, inplace=True)),

                    ("conv2", spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))),
                    ("lrelu2", nn.LeakyReLU(0.2, inplace=True)),

                    ("conv3", spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))),
                    ("lrelu3", nn.LeakyReLU(0.2, inplace=True)),

                    ("conv4", spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))),
                    ("lrelu4", nn.LeakyReLU(0.2, inplace=True)),
                ]
            )
        )
        self.fc = nn.Linear((ndf * 8) * 8 * 8, 1)

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, (self.ndf * 8) * 8 * 8)
        output = torch.sigmoid(self.fc(x))
        return output.squeeze()
