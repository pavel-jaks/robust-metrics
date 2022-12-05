from itertools import product

import torch
import torchvision
import skimage.metrics as skit

import metrics


def get_mnist_pair():
    mnist = torchvision.datasets.MNIST('mnist', download=True, transform=torchvision.transforms.ToTensor())
    image = mnist[0][0]
    
    noised_image = (image + (0.1 * torch.rand(image.shape))).apply_(lambda x: 0 if x < 0 else 255 if x > 255 else x)
    return image, noised_image

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

test(True)
