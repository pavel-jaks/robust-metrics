from itertools import product
from math import sqrt

import torch
import torchvision
import skimage.metrics as skit
from geomloss import SamplesLoss

import metrics
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, linprog

def get_mnist_pair():
    mnist = torchvision.datasets.MNIST('mnist', download=True, transform=torchvision.transforms.ToTensor())
    image = mnist[0][0]
    
    noised_image = (image + (0.1 * torch.rand(image.shape))).apply_(lambda x: 0 if x < 0 else 255 if x > 255 else x)
    return image, noised_image

def get_mnist_two_images():
    mnist = torchvision.datasets.MNIST('mnist', download=True, transform=torchvision.transforms.ToTensor())
    return mnist[0][0].unsqueeze(0), mnist[1][0].unsqueeze(0)

def wasserstein_linear_program(x, y, cost_matrix):
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("Not an image")
    if x.shape != y.shape:
        raise ValueError("Given images of different shapes")
    
    # batch = x.shape[0]
    # channels = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]

    x_vector = x.flatten()
    y_vector = y.flatten()
    x_vector = x_vector / x_vector.sum()
    y_vector = y_vector / y_vector.sum()

    n = width * height

    Ap, Aq = [], []
    z = np.zeros((n, n))
    z[:, 0] = 1

    for i in range(n):
        Ap.append(z.ravel())
        Aq.append(z.transpose().ravel())
        z = np.roll(z, 1, axis=1)

    A = np.row_stack((Ap, Aq))[:-1]
    b = np.concatenate((x_vector, y_vector))[:-1]

    result = linprog(cost_matrix.ravel(), A_eq=A, b_eq=b)
    return np.sum(result.x.reshape((n, n)) * cost_matrix.numpy())

def test(verbose=False):
    image, noised_image = get_mnist_pair()
    
    mse_ref = skit.mean_squared_error(image.numpy(), noised_image.numpy())
    mse_my = metrics.MeanSquaredError()(image.unsqueeze(0), noised_image.unsqueeze(0))
    assert abs(mse_ref - mse_my) < 1e-5
    if verbose:
        print(f'MSE:{mse_ref};MyMSE:{mse_my}')
    
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
    ssim_my = metrics.StructuralSimilarityIndexMeasure(window_size=7, l=1)(image.unsqueeze(0), noised_image.unsqueeze(0))
    assert abs(ssim_ref - ssim_my) < 1e-5
    if verbose:
        print(f'SSIM:{ssim_ref};MySSIM:{ssim_my}')

    psnr_ref = skit.peak_signal_noise_ratio(image.numpy(), noised_image.numpy(), data_range=1)
    psnr_my = metrics.PeakSignalToNoiseRatio()(image.unsqueeze(0), noised_image.unsqueeze(0))
    assert abs(psnr_ref - psnr_my) < 1e-5
    if verbose:
        print(f'PSNR:{psnr_ref};MyPSNR:{psnr_my}')

    
    # wasserstein_my = metrics.WassersteinApproximation()(image.unsqueeze(0), noised_image.unsqueeze(0))
    # image_one, image_two = get_mnist_two_images()
    width, height = 5, 5
    image_one = np.random.rand(width, height)
    image_two = np.random.rand(width, height)
    image_one /= np.sum(image_one)
    image_two /= np.sum(image_two)
    image_one = torch.tensor(image_one, dtype=torch.float32)
    image_two = torch.tensor(image_two, dtype=torch.float32)
    image_one, image_two = image_one.reshape(1, 1, width, height), image_two.reshape(1, 1, width, height)

    regularization = 35
    iterations = 300
    wasserstein_my = metrics.WassersteinApproximation(
        regularization=regularization,
        iterations=iterations
    )(image_one, image_two)
    
    # ref = SamplesLoss(
    #     loss="sinkhorn",
    #     p=1,
    #     blur= (regularization)
    # )

    # width, height = 5, 5
    cost_matrix = torch.tensor(
            [
                [
                    sqrt((i // width - j // width) ** 2 + (i % width - j % width) ** 2) 
                    for j in range(width * height)
                ]
                for i in range(width * height)
            ]
        )

    # x, y = image_one.flatten(), image_two.flatten()
    # x_norm, y_norm = (x / x.sum(dim=(2, 3), keepdim=True)).reshape(batch, width * height), \
        # (y / y.sum(dim=(2, 3), keepdim=True)).reshape(batch, width * height)
    
    ref_value = wasserstein_linear_program(image_one, image_two, cost_matrix)

    if verbose:
        print(f'Wasserstein:{ref_value};MyWasserstein:{wasserstein_my}')

test(True)
