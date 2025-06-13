import torch
import torch.nn as nn
from math import floor

class Reparameterize(nn.Module):
    """
    Module to perform the reparameterization trick for the VAE.

    This enables gradient backpropagation through the sampling operation
    by sampling epsilon and transforming it using the mean and log variance.
    """
    def forward(self, mean, log_var):
        """
        Applies the reparameterization trick: z = mean + std * epsilon

        Args:
            mean (Tensor): Mean of the latent Gaussian distribution.
            log_var (Tensor): Log-variance of the latent Gaussian distribution.

        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std


class LatentGaussian(nn.Module):
    """
    Maps encoded feature maps to the parameters of a latent Gaussian distribution.
    """
    def __init__(self, latent_size):
        """
        Initializes mean and log variance projection layers.

        Args:
            latent_size (int): Size of the latent vector.
        """
        super().__init__()
        self.latent_size = latent_size
        self.mean_layer = nn.LazyLinear(latent_size)
        self.log_var_layer = nn.LazyLinear(latent_size)

    def forward(self, z):
        """
        Computes mean and log variance from input tensor.

        Args:
            z (Tensor): Flattened encoder output.

        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance.
        """
        z = z.view(z.size(0), -1)
        mean = self.mean_layer(z)
        log_var = self.log_var_layer(z)
        return mean, log_var


class Encoder(nn.Module):
    """
    Convolutional encoder network that maps input images to a latent distribution.
    """
    def __init__(self, input_channels, latent_size, feature_sizes=[64, 128, 256, 512]):
        """
        Initializes convolutional layers and latent Gaussian projection.

        Args:
            input_channels (int): Number of channels in input image.
            latent_size (int): Size of the latent vector.
            feature_sizes (List[int]): List of convolution feature sizes.
        """
        super().__init__()

        layers = [
            nn.Conv2d(input_channels, feature_sizes[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_sizes[0]),
            nn.LeakyReLU()
        ]

        for i in range(1, len(feature_sizes)):
            layers.append(nn.Conv2d(feature_sizes[i - 1], feature_sizes[i], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(feature_sizes[i]))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Conv2d(feature_sizes[-1], latent_size, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(latent_size))

        self.layers = nn.Sequential(*layers)
        self.gaussian = LatentGaussian(latent_size)
        self.reparameterize = Reparameterize()

    def forward(self, x):
        """
        Encodes input image into latent vector.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Sampled latent vector, mean, and log variance.
        """
        x = self.layers(x)
        mean, log_var = self.gaussian(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var


class Decoder(nn.Module):
    """
    Convolutional decoder network that reconstructs images from latent vectors.
    """
    def __init__(self, output_channels, latent_size, input_shape, feature_sizes=[64, 128, 256, 512][::-1]):
        """
        Initializes transpose convolutional layers for upsampling.

        Args:
            output_channels (int): Number of output channels (e.g., 3 for RGB).
            latent_size (int): Size of the latent vector.
            input_shape (Tuple[int, int, int]): Shape of the input image.
            feature_sizes (List[int]): List of transpose convolution feature sizes.
        """
        super().__init__()

        self.input_shape = input_shape
        self.feature_sizes = feature_sizes

        self.bottleneck_size = (
            feature_sizes[0],
            floor(input_shape[1] / 2 ** len(feature_sizes)),
            floor(input_shape[2] / 2 ** len(feature_sizes))
        )

        self.upsample = nn.Linear(latent_size, self.bottleneck_size[0] * self.bottleneck_size[1] * self.bottleneck_size[2])

        layers = []
        for i in range(len(feature_sizes) - 1):
            layers.append(nn.ConvTranspose2d(feature_sizes[i], feature_sizes[i + 1], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(feature_sizes[i + 1]))
            layers.append(nn.LeakyReLU())

        layers.append(nn.ConvTranspose2d(feature_sizes[-1], output_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        """
        Reconstructs image from latent vector.

        Args:
            z (Tensor): Latent vector.

        Returns:
            Tensor: Reconstructed image.
        """
        batch = z.shape[0]
        x = self.upsample(z)
        x = x.view(batch, *self.bottleneck_size)
        x = self.layers(x)
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder class combining encoder and decoder networks.
    """
    def __init__(self, input_shape, latent_size, feature_sizes=[64, 128, 256, 512]):
        """
        Initializes the encoder and decoder components of the VAE.

        Args:
            input_shape (Tuple[int, int, int]): Shape of input image.
            latent_size (int): Size of the latent vector.
            feature_sizes (List[int]): Convolutional feature sizes.
        """
        super().__init__()

        self.input_size = input_shape
        self.latent_size = latent_size
        self.feature_sizes = feature_sizes

        self.encoder = Encoder(
            input_channels=input_shape[0],
            latent_size=latent_size,
            feature_sizes=feature_sizes
        )

        self.decoder = Decoder(
            output_channels=input_shape[0],
            feature_sizes=feature_sizes[::-1],
            latent_size=latent_size,
            input_shape=input_shape
        )

    def forward(self, x):
        """
        Runs the input through the VAE to get reconstruction and latent stats.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Reconstructed image, mean, and log variance.
        """
        z, mean, log_var = self.encoder(x)
        z = self.decoder(z)
        return z, mean, log_var

    def decode(self, z):
        """
        Decodes a latent vector into an image.

        Args:
            z (Tensor): Latent vector.

        Returns:
            Tensor: Reconstructed image.
        """
        return self.decoder(z)