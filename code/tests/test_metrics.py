from typing import Tuple

import pytest

import torch
import torchvision
import skimage.metrics as skit
import numpy as np
import cvxpy as cvx

import metrics


def get_mnist_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for retrieving a pair of a random MNIST image and its' corrupted version

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



def test_mse() -> None:
    for _ in range(10):
        image, noised_image = get_mnist_pair()
        
        # MSE testing
        mse_ref = skit.mean_squared_error(image.numpy(), noised_image.numpy())
        mse_my = metrics.MeanSquaredError()(image.unsqueeze(0), noised_image.unsqueeze(0))
        assert abs(mse_ref - mse_my) < 1e-5

def test_dssim():
    for _ in range(10):
        image, noised_image = get_mnist_pair()
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

def test_psnr():
    for _ in range(10):
        image, noised_image = get_mnist_pair()

        # PSNR testing
        psnr_ref = skit.peak_signal_noise_ratio(image.numpy(), noised_image.numpy(), data_range=1)
        psnr_my = metrics.PeakSignalToNoiseRatio()(image.unsqueeze(0), noised_image.unsqueeze(0))
        assert abs(psnr_ref - psnr_my) < 1e-5

def test_wasserstein():
    for i in range(10):
        print(f"Run {i + 1} of Wasserstein")
        width = height = 5
        image1, image2 = np.random.rand(1, 1, width, height), np.random.rand(1, 1, width, height)
        # image1 = image2 = np.random.rand(c,c)

        torch_im1, torch_im2 = torch.Tensor(image1), torch.Tensor(image2)

        regularization = 5
        wasserstein_ref = ref_wasserstein(
            torch_im1,
            torch_im2,
            regularization
        )
        wasserstein_my = metrics.WassersteinApproximation(regularization=regularization, iterations=25000, tolerance=1e-30, division_const=1e-20)(torch_im1, torch_im2)
        print(f"Ref={wasserstein_ref}; My={wasserstein_my.item()}")
        assert abs(wasserstein_my - wasserstein_ref) < 1e-5

def ref_wasserstein(image1: np.ndarray, image2: np.ndarray, regularization):
    _, _, width, height = image1.shape
    wh = width * height
    wwhh = wh ** 2

    cost_matrix = np.array(
        [
            [
                abs(i // width - j // width) + abs(i % width - j % width) 
                for j in range(width * height)
            ]
            for i in range(width * height)
        ]
    )
    image1_vector, image2_vector = image1.flatten(), image2.flatten()
    image1_vector, image2_vector = image1_vector / image1_vector.sum(), image2_vector / image2_vector.sum()

    # print(image1_vector)
    # print(image2_vector)

    # positive_constraint = opt.LinearConstraint(np.eye(width * width * height * height), 0, keep_feasible=True)

    x = cvx.Variable(wwhh)

    objective = cvx.Minimize(cvx.sum(cvx.multiply(cost_matrix.flatten(), x)) - (1 / regularization) * cvx.sum(cvx.entr(x)))
    constraints = [
        x >= 0,
        x.reshape((wh, wh)).sum(0) == image1_vector,
        x.reshape((wh, wh)).sum(1) == image2_vector
    ]

    problem = cvx.Problem(objective, constraints)
    _ = problem.solve(solver=cvx.ECOS)
    
    return (x.value * cost_matrix.flatten()).sum()
