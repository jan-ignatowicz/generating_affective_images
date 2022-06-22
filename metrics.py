import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


# Counting FID and KID functions
# (https://github.com/mseitzer/pytorch-fid)

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


model = InceptionV3()
model.eval()


def compute_kid(real, fake, batch_size=128):
    model.to(real.device)
    if real.size()[1] == 1:
        real = torch.cat((real, real, real), dim=1)
        fake = torch.cat((fake, fake, fake), dim=1)
    print("Computing Inception Activations")
    real_activations = get_activations(real, model, batch_size)
    fake_activations = get_activations(fake, model, batch_size)
    print("Computing KID")
    return _kid(real_activations, fake_activations)


def compute_fid(real, fake, batch_size=128):
    model.to(real.device)
    if real.size()[1] == 1:
        real = torch.cat((real, real, real), dim=1)
        fake = torch.cat((fake, fake, fake), dim=1)

    real_activations = get_activations(real, model, batch_size)
    real_mu = np.mean(real_activations, axis=0)
    real_sigma = np.cov(real_activations, rowvar=False)

    fake_activations = get_activations(fake, model, batch_size)
    fake_mu = np.mean(fake_activations, axis=0)
    fake_sigma = np.cov(fake_activations, rowvar=False)

    return _fid(real_mu, real_sigma, fake_mu, fake_sigma)


def compute_metrics(real, fake, batch_size=128):
    model.to(real.device)
    if real.size()[1] == 1:
        real = torch.cat((real, real, real), dim=1)
        fake = torch.cat((fake, fake, fake), dim=1)
    real_activations = get_activations(real, model, batch_size)
    fake_activations = get_activations(fake, model, batch_size)

    kid = _kid(real_activations, fake_activations)

    real_mu = np.mean(real_activations, axis=0)
    real_sigma = np.cov(real_activations, rowvar=False)
    fake_mu = np.mean(fake_activations, axis=0)
    fake_sigma = np.cov(fake_activations, rowvar=False)
    fid = _fid(real_mu, real_sigma, fake_mu, fake_sigma)

    return kid, fid


def get_activations(images, model, batch_size=128, dims=2048):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """

    loader = DataLoader(images, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing inception activations"):
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            preds.append(pred)
        preds = torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()
    return preds


def _kid(X, Y):
    """
    Given X, Y (numpy) batches of inception outputs of generated and real images,
    return the Kernel Inception Distance. X and Y have to have the same dimensions.
    """
    print(X.shape)
    print(Y.shape)
    assert np.all(X.shape == Y.shape)

    n = X.shape[0]

    def k(x, y):
        # Kernel
        return (1 / x.shape[0] * np.dot(x, y) + 1) ** (1 / 3)

    def f(X, Y):
        # First 2 sums. We use the fact that k is symmetric
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                res += k(X[i], Y[j])

        return 2 * res

    def f2(X, Y):
        # Third sum.
        res = f(X, Y)

        for i in range(n):
            res += k(X[i], Y[i])

        return res

    return 1 / (n * (n - 1)) * f(X, X) + 1 / (n * (n - 1)) * f(Y, Y) - 2 / (n ** 2) * f2(X, Y)


def _fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
