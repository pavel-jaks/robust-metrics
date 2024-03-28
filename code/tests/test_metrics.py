from typing import Tuple

import pytest

import torch
import torchvision
import skimage.metrics as skit
import numpy as np
import scipy.optimize as opt

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

        regularization = 7
        wasserstein_ref = wasserstein_linear_program(
            torch_im1,
            torch_im2,
            cost_matrix=torch.Tensor(
                [
                    [
                        abs(i // width - j // width) + abs(i % width - j % width) 
                        for j in range(width * height)
                    ]
                    for i in range(width * height)
                ]
            )
        )
        wasserstein_my = metrics.WassersteinApproximation(regularization=regularization, iterations=25000, tolerance=1e-30, division_const=1e-20)(torch_im1, torch_im2)
        print(f"Ref={wasserstein_ref}; My={wasserstein_my.item()}")
        assert abs(wasserstein_my - wasserstein_ref) < 1e-2

def ref_wasserstein(image1: np.ndarray, image2: np.ndarray, regularization):
    width, height = image1.shape
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
    probability_constraint1 = opt.NonlinearConstraint(
        lambda x: x.reshape(width * height, width * height).sum(0),
        image1_vector,
        image1_vector
    )
    probability_constraint2 = opt.NonlinearConstraint(
        lambda x: x.reshape(width * height, width * height).sum(1),
        image2_vector,
        image2_vector
    )
    objective = lambda x: x.dot(cost_matrix.flatten()) - (1 / regularization) * (- x[x > 0] * np.log(x[x > 0])).sum()
    init = (image2_vector.reshape(width * height, 1) @ image1_vector.reshape(1, width * height)).flatten() + np.random.randn(width * height * width * height) / (width ** 2 * height ** 2)
    # init = np.ones(width ** 2 * height ** 2) + (1 / 3) * np.random.randn(width * height * width * height)
    # init = abs(init)
    # init /= init.sum()
    init[init < 1e-3] = 1e-3
    init[init > 1 - 1e-3] = 1 - 1e-3

    

    # print(init)
    initial = np.copy(init)

    bounds = opt.Bounds(np.zeros((width ** 2 * height ** 2)), np.ones((width ** 2 * height ** 2)))

    opt_res = opt.minimize(
        objective,
        initial,
        method="trust-constr",
        jac=lambda x: cost_matrix.flatten() + 1 / regularization + (np.array([z for z in map(lambda y: 0 if y <= 0 else np.log(y), x)])) / regularization,
        hess=lambda x: np.diag((1 / regularization) / x),
        constraints=(probability_constraint1, probability_constraint2),
        bounds=bounds
    )
    # print(opt_res.x)
    return opt_res.x.dot(cost_matrix.flatten())

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

    result = opt.linprog(cost_matrix.ravel(), A_eq=A, b_eq=b)
    return np.sum(result.x.reshape((n, n)) * cost_matrix.numpy())
