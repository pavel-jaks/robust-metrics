# Metrics module documentation

## File metrics.py

The file [metrics.py](metrics.py) contains the implementation of various metrics of visual similarity
using the [PyTorch](https://pytorch.org/) library.

### Class Transform

Class derived from `torch.nn.Module`.
Base class to encapsulate transformation of data applied by a metric of visual similarity before computing the actual distance.

* Empty class

### Class Identity

Class derived from `Transform` class to perform identity transformation. The instance of this class is later used as a default value for transform parametr in the Metric class.

* Overrides `torch.nn.Module`'s `forward` function to return its' only `torch.Tensor` parameter.

### Class Metric

Class that is a base class for all classes that implement a metric of visual similarity.
Derived from `torch.nn.Module`.

* Parameters of the constructor:
  * Parameter `transform` of type `Transform`.
* Overrides `torch.nn.Module`'s `forward` method to apply a `Transform` passed in the constructor to its' two parameters of type `torch.Tensor` and then calls `compute` with transformed `Tensor`s as parameters to compute their metric of visual similarity.
  * Returns batch of distances stored in `torch.Tensor`.
* Defines abstract method `compute` which computes the actual distances between two batches of images that are passed as `torch.Tensor` objects.
  * Returns batch of distances stored in `torch.Tensor`.

### Class Norm

Abstract class derived from `torch.nn.Module` which encapsulates a norm.
Its' usage is then implementation of a metric that is derived from a norm.

* Forces types derived from a `Norm` to implement `forward` method in order to compute the norm of `torch.Tensor` passed as the only parameter to the method.

### Class LpNorm

Implementation of a mathematical $l_p$ vector norm.
Derived from `Norm`.

* Parameters of the contructor:
  * Parameter `p` of type `int` that decides which $l_p$ norm to compute. Must be greater than or equal to 1.
* Implements `forward` method.
  * Takes these arguments:
    * Argument `x` of type `torch.Tensor` with expected `ndim` at least 2, where the first dimension is for batch.
  * Returns a `torch.Tensor` object that contains a batch of norms.

### Class L2Norm

Special case of `LpNorm` with parameter `p` set to `2`.

### Class L1Norm

Special case of `LpNorm` with parameter `p` set to `1`.

### Class LinfNorm

Derived from the `Norm` class, `LinfNorm` is the implementation of $l_\infty$ vector norm.

* Overrides `forward` method to perform computation.
  * Its' arguments:
    * Parameter `x` of type `torch.Tensor` containing the batch of images of which the norm shall be computed.
  * Returns `torch.Tensor` object with batch of norms computed.

### Class L0Norm

Derived from `Norm`, `L0Norm` provides a way of computing $l_0$ pseudo-norm.

Warning: This pseudo-norm is not automatically differentiable in PyTorch sense!

* Overrides `forward` method to perform computation.
  * Its' arguments:
    * Parameter `x` of type `torch.Tensor` containing the batch of images of which the pseudo-norm shall be computed.
  * Returns `torch.Tensor` object with batch of pseudo-norms computed.

### Class MetricFromNorm

Derived from `Metric`, `MetricFromNorm` encapsulates a metric distance that is computed as a norm of a difference of two tensors.

* Constructor arguments:
  * Parameter `norm` of type `Norm` which is the underlying norm to compute the metric from.
  * Parameter `transform` of type `Transform` or `None` which is passed to the constructor of the base class `Metric`.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

### Class LpMetric

Special case of `MetricFromNorm` class which uses a `LpNorm` class to compute the metrics of two batches of images.

* Constructor takes these arguments:
  * Parameter `p` of type `int` to passed to the constructor of `LpNorm`.
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.

### Class L2Metric

Special case of `LpMetric` class which uses a `LpNorm` class with parameter `p` set to `2` to compute the metrics of two batches of images.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.

### Class L0Metric

Special case of `MetricFromNorm` class which uses a `L0Norm` class to compute the metrics of two batches of images.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.

### Class LinfMetric

Special case of `MetricFromNorm` class which uses a `LinfNorm` class to compute the metrics of two batches of images.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.

### Class MeanSquaredError

Special case of `Metric` class which computes the metrics of two batches of images as a mean squared error.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

### Class RootMeanSquaredError

Special case of `Metric` class which computes the metrics of two batches of images as a root mean squared error does.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

### Class PeakSignalToNoiseRatio

Special case of `Metric` class which computes the metrics of two batches of images PSNR (peak signal to nois ratio) does.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.
  * Parameter `l` of type `float` which is the dynamical range of the images passed to compute their distances.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

### Class NoiseToPeakSignalRatio

Special case of `Metric` class which computes the metrics of two batches of images negative of PSNR (peak signal to noise ratio).

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.
  * Parameter `l` of type `float` which is the dynamical range of the images passed to compute their distances.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

### Class StructuralDissimilarity

Special case of `Metric` class which computes the metrics of two batches of images as a structural dissimilarity does.
It computes the index SSIM (structural similarity index measure) and then computes DSSIM (structural dissimilarity) as (1 - SSIM) / 2.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.
  * Parameter `window_size` which should always be a positive integer. Decides the size of a sliding window in which the SSIMs are computed and then averaged.
  * Parameter `k_1` to compute the first constant used for division safety.
  * Parameter `k_2` to compute the second constant used for division safety.
  * Parameter `l` is the peak signal/ dynamical range of the images.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

### Enum CostMatrixType

Enum to handle the type of a cost matrix used in the computation of the approximation of Wasserstein distance.

### Class WassersteinApproximation

Special case of `Metric` class which computes the approxiamtion of Wasserstein distances of two batches of images. It computes the dual Sinkhorn divergence using the Sinkhorn fixed point iterations.

* Constructor takes these arguments:
  * Parameter `transform` of type `Transform` or `None` to be passed to its' base class constructor.
  * Parameter `regularization` of type `float` of which reciprocal value is used as a coefficient with which to subtract from optimal transport problem objective function the entropic penalty.
  * Parameter `iterations` of type `int` which sets the maximal number of iterations used to compute the distance.
  * Parameter `division_const` of type `float` to alter the small numbers by which shall be divided to be at least a bit bigger.
  * Parameter `cost_matrix_type` of type `CostMatrixType` which decides the underlying vector distance.
  * Parameter `tolerance` of type `float` to be used as a possible stopping criterion.
* Implements `compute` method to compute the distances of two batches of images both stored in `torch.Tensor` object.
  * Returns `torch.Tensor` object that contains a batch of distances.

## Usage

```Python
# import of PyTorch library
import torch
# import of desired metric
from metrics import WassersteinApproximation

# random images (scaled to [0, 1] interval)
# tensor of shape: Batch x channels x width x height
image1 = torch.rand(1, 1, 20, 20)
image2 = torch.rand(1, 1, 20, 20)
# metric instantiation
metric = WassersteinApproximation(regularization=5)

# the actual metric computation
distance = metric(image1 , image2)
```
