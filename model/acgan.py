import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False, dropout=False,
               sn=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    elif sn:
        module.append(spectral_norm(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn)))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    if dropout:
        module.append(nn.Dropout(0.2))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, latent_dim=10, num_classes=10, label_embed_size=5, num_channels=3,
                 ngf=64):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embed_size)
        self.tconv1 = conv_block(latent_dim + label_embed_size, ngf * 4, pad=0, transpose=True)
        self.tconv2 = conv_block(ngf * 4, ngf * 2, transpose=True)
        self.tconv3 = conv_block(ngf * 2, ngf, transpose=True)
        self.tconv4 = conv_block(ngf, num_channels, k_size=4, stride=4, pad=0, transpose=True,
                                 use_bn=False)

    def forward(self, x, label):
        x = x.reshape([x.shape[0], -1, 1, 1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape([label_embed.shape[0], -1, 1, 1])
        x = torch.cat((x, label_embed), dim=1)

        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))

        x = torch.tanh(self.tconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, n_classes=10, num_channels=3, conv_dim=64):
        super(Discriminator, self).__init__()
        self.image_size = 64
        self.label_embedding = nn.Embedding(n_classes, self.image_size * self.image_size)

        self.conv1 = conv_block(num_channels + 1, conv_dim, use_bn=False, dropout=True)
        self.conv2 = conv_block(conv_dim, conv_dim * 2, dropout=True)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4, dropout=True)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=4, stride=6, pad=0, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        alpha = 0.2
        label_embed = self.label_embedding(label)

        label_embed = label_embed.reshape(
            [label_embed.shape[0], 1, self.image_size, self.image_size])

        x = torch.cat((x, label_embed), dim=1)
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = torch.sigmoid(self.conv4(x))

        return x.squeeze()


class DiscriminatorDropout(nn.Module):
    def __init__(self, n_classes=10, num_channels=3, conv_dim=64):
        super(DiscriminatorDropout, self).__init__()
        self.image_size = 64
        self.label_embedding = nn.Embedding(n_classes, self.image_size * self.image_size)

        self.conv1 = conv_block(num_channels + 1, conv_dim, use_bn=False, dropout=True)
        self.conv2 = conv_block(conv_dim, conv_dim * 2, dropout=True)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4, dropout=True)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=4, stride=6, pad=0, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        alpha = 0.2
        label_embed = self.label_embedding(label)

        label_embed = label_embed.reshape(
            [label_embed.shape[0], 1, self.image_size, self.image_size])

        x = torch.cat((x, label_embed), dim=1)
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = torch.sigmoid(self.conv4(x))

        return x.squeeze()


class DiscriminatorSN(nn.Module):
    def __init__(self, n_classes=10, num_channels=3, conv_dim=64):
        super(DiscriminatorSN, self).__init__()
        self.image_size = 32
        self.label_embedding = nn.Embedding(n_classes, self.image_size * self.image_size)
        self.conv1 = conv_block(num_channels + 1, conv_dim, use_bn=False, sn=True)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=4, stride=1, pad=0, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        alpha = 0.2
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape(
            [label_embed.shape[0], 1, self.image_size, self.image_size])
        x = torch.cat((x, label_embed), dim=1)
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = torch.sigmoid(self.conv4(x))
        return x.squeeze()
