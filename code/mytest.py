from math import sqrt
from typing import Tuple

import torch
import torchvision
import skimage.metrics as skit
import geomloss
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, linprog

import metrics


def get_mnist_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for retrieving a pair of MNIST image and its' corrupted version

    :returns: pair of MNIST image and its' corrupted version; scaled to [0, 1]
    """
    mnist = torchvision.datasets.MNIST('mnist', download=True, transform=torchvision.transforms.ToTensor())
    image = mnist[np.random.randint(0,10000)][0]
    
    noised_image = (image + (0.1 * torch.rand(image.shape))).apply_(lambda x: 0 if x < 0 else 1 if x > 1 else x)
    return image, noised_image


def get_mnist_two_images() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for retrieving a pair of MNIST images

    :returns: pair of MNIST images; scaled to [0, 1]
    """
    mnist = torchvision.datasets.MNIST('mnist', download=True, transform=torchvision.transforms.ToTensor())
    ix = np.random.randint(0,10000)
    return mnist[ix][0].unsqueeze(0), mnist[ix + 1][0].unsqueeze(0)


def wasserstein_linear_program(x: torch.Tensor, y: torch.Tensor, cost_matrix: torch.Tensor) -> float:
    """
    Function to compute precise 1-Wasserstein distance (using SciPy linear programming interface)
    Currently works only for single channel images and only for one in a batch.

    :param x: First argument of the metric (shape = (batch, channels, width, height))
    :param y: Second argument of the metric (shape = (batch, channels, width, height))
    :param cost_matrix: Cost matrix of the linear program.

    :returns: The 1-Wasserstein distance between x and y.
    """
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("Param x or y is in unsupported shape.")
    if x.shape != y.shape:
        raise ValueError("Given images of different shapes.")
    
    batch = x.shape[0]
    channels = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]

    if batch != 1:
        raise NotImplementedError("Given more images in a batch, not supported yet.")
    if channels != 1:
        raise NotImplementedError("Given multi channel images, not supported yet.")

    x_vector = x.flatten()
    y_vector = y.flatten()
    x_vector = x_vector / x_vector.sum()
    y_vector = y_vector / y_vector.sum()

    n = width * height

    Ap, Aq = [], []
    z = np.zeros((n, n))
    z[:, 0] = 1

    for _ in range(n):
        Ap.append(z.ravel())
        Aq.append(z.transpose().ravel())
        z = np.roll(z, 1, axis=1)

    A = np.row_stack((Ap, Aq))[:-1]
    b = np.concatenate((x_vector, y_vector))[:-1]

    result = linprog(cost_matrix.ravel(), A_eq=A, b_eq=b)
    return np.sum(result.x.reshape((n, n)) * cost_matrix.numpy())


def transform_image_for_geomloss(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("Param x or y is in unsupported shape.")

    batch = x.shape[0]
    channels = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]

    if batch != 1:
        raise NotImplementedError("Given more images in a batch, not supported yet.")
    if channels != 1:
        raise NotImplementedError("Given multi channel images, not supported yet.")

    pixels = x * 255
    pixels = pixels.reshape(width, height)
    cloud = []
    for i in range(width):
        for j in range(height):
            for k in range(int(pixels[i, j])):
                cloud.append([i , j])
    return torch.tensor(cloud)


def dual_sinkhorn(x: torch.Tensor, y: torch.Tensor, regularization: float) -> float:
    ds_metric = geomloss.SamplesLoss(
        loss="sinkhorn",
        p = 1,
        blur = 1 / regularization
    )
    x_transformed = transform_image_for_geomloss(x)
    y_transformed = transform_image_for_geomloss(y)
    return ds_metric(x_transformed, y_transformed)


def test(verbose=False) -> None:
    """
    Asserts correctness of metrics implemented in metrics.py using different libraries as references.

    :param verbose: Whether to print metric results.

    :returns: None; if it succeeds, no error is raised.
    """
    image, noised_image = get_mnist_pair()
    
    # MSE testing
    mse_ref = skit.mean_squared_error(image.numpy(), noised_image.numpy())
    mse_my = metrics.MeanSquaredError()(image.unsqueeze(0), noised_image.unsqueeze(0))
    assert abs(mse_ref - mse_my) < 1e-5
    if verbose:
        print(f'MSE:{mse_ref};MyMSE:{mse_my}')
    
    # DSSIM testing
    ssim_ref = skit.structural_similarity(
        image.numpy(),
        noised_image.numpy(),
        win_size=7,
        data_range=1,
        channel_axis=0,
        use_sample_covariance=True,
        K1=1e-2,
        K2=3e-2
    )
    dssim_ref = (1 - ssim_ref) / 2
    dssim_my = metrics.StructuralDissimilarity(window_size=7, l=1)(image.unsqueeze(0), noised_image.unsqueeze(0))
    assert abs(dssim_ref - dssim_my) < 1e-5
    if verbose:
        print(f'SSIM:{dssim_ref};MySSIM:{dssim_my}')

    # PSNR testing
    psnr_ref = skit.peak_signal_noise_ratio(image.numpy(), noised_image.numpy(), data_range=1)
    psnr_my = metrics.PeakSignalToNoiseRatio()(image.unsqueeze(0), noised_image.unsqueeze(0))
    assert abs(psnr_ref - psnr_my) < 1e-5
    if verbose:
        print(f'PSNR:{psnr_ref};MyPSNR:{psnr_my}')

    
    # Wasserstein testing
    regularization = 5
    wasserstein_ref = dual_sinkhorn(image.unsqueeze(0), noised_image.unsqueeze(0), regularization)
    wasserstein_my = metrics.WassersteinApproximation(
        regularization=regularization,
        cost_matrix_type=metrics.CostMatrixType.L2
    )(image.unsqueeze(0), noised_image.unsqueeze(0))

    if verbose:
        print(f'Wasserstein:{wasserstein_ref};MyWasserstein:{wasserstein_my}')

if __name__ == '__main__':
    test(True)
