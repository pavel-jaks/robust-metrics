from enum import Enum
from typing import Union
from abc import abstractmethod
import itertools
from math import sqrt

import torch
import torch.nn as nn


class Transform(nn.Module):
    """
    Class to encapsulate tranformation of data
    """

    pass


class Identity(Transform):
    """
    Identity transformation
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Metric(nn.Module):
    """
    Module that encapsulates implpementation of a mathematical concept of metric
    With possibility of a tranformation applied
    """

    def __init__(self, transform: Union[Transform, None] = None):
        """
        Constructor
        :param transform: Transformation to be applied
        """
        super().__init__()
        if transform is None:
            self.transform = Identity()
        else:
            self.transform = transform

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        transformed_x, transformed_y = self.transform(x), self.transform(y)
        return self.compute(transformed_x, transformed_y)

    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Actual computation of the metric distance
        :param x: input No. I
        :param y: input No. II
        :return: Tensor of batch of metrics
        """
        pass


class Norm(nn.Module):
    """
    Encapsulation of a norm
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of which norm shall be computed (shape: Batch x the_rest)

        :return: Tensor of batch of norms
        """
        pass


class LpNorm(Norm):
    """
    Implementation of an L_p norm for a given positive integer p
    """

    def __init__(self, p: int):
        """
        :param p: positive integer
        """
        super().__init__()
        if p < 1:
            raise ValueError("p must be greater than 1")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of which norm shall be computed (shape: Batch x the_rest)

        :return: Tensor of batch of norms
        """
        return (x.abs() ** self.p).sum(dim=tuple(range(1, x.ndim))) ** (1 / self.p)


class L2Norm(LpNorm):
    """
    Special case of an LpNorm - L_2
    """
    
    def __init__(self):
        super().__init__(2)


class L1Norm(LpNorm):
    """
    Special case of an LpNorm - L_1
    """
    def __init__(self):
        super().__init__(1)


class LinfNorm(Norm):
    """
    Implementation of an L_infty norm
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of which norm shall be computed (shape: Batch x the_rest)

        :return: Tensor of batch of norms
        """
        out = x.abs()
        for _ in range(1, out.ndim):
            out = out.max(dim=1)[0]
        return out


class L0Norm(Norm):
    """
    Implementation of a L_0 norm
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of which norm shall be computed (shape: Batch x the_rest)

        :return: Tensor of batch of norms
        """

        return ((x != torch.zeros(x.shape)) * 1).sum(dim=tuple(range(1, x.ndim)))


class MetricFromNorm(Metric):
    """
    Encapsulation of metric distance which is derived as a norm of a difference
    """

    def __init__(self, norm: Norm, transform: Union[Transform, None] = None):
        """
        Implementation of a constructor
        :param norm: Norm
        :param transform: Transformation to be applied
        """
        super().__init__(transform)
        self.norm = norm

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the actual computation using the given norm
        :param x: Tensor x
        :param y: Tensor y
        :return: Tensor of batch of metrics
        """
        out = self.norm(x - y)
        return out


class LpMetric(MetricFromNorm):
    """
    Metric devived from a L_p norm
    """
    
    def __init__(self, p: int, transform: Union[Transform, None] = None):
        """
        Implementation of a constructor
        :param p: Positive integer p
        :param transform: Transformation to be applied
        """
        super().__init__(LpNorm(p), transform)


class L2Metric(LpMetric):
    """
    Implementation of a L_2 metric derived from L_2 norm
    """
    
    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(2, transform)


class L0Metric(MetricFromNorm):
    """
    Implementation of a L_0 metric derived from L_0 norm
    """

    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(L0Norm(), transform)


class LinfMetric(MetricFromNorm):
    """
    Implementation of a L_infty metric derived from L_infty norm
    """

    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(LinfNorm(), transform)


class MeanSquaredError(Metric):
    """
    Implementation of MSE as a metric
    """

    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(transform)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((x - y) ** 2).mean(dim=tuple(range(1, x.ndim)))


class RootMeanSquaredError(Metric):
    """
    Implementation of RMSE as a metric
    """

    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(transform)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((x - y) ** 2).mean(dim=tuple(range(1, x.ndim))) ** (1 / 2)


class PeakSignalToNoiseRatio(Metric):
    """
    Implementation of PSNR
    """

    def __init__(self, transform: Union[Transform, None] = None, l: float = 1):
        """
        Constructor

        :param l: peak signal of the image
        """
        super().__init__(transform)
        self.l = l
        self.rmse = RootMeanSquaredError()

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = 20 * torch.log10(self.l / self.rmse(x, y))
        return out
    
class NoiseToPeakSignalRatio(Metric):
    """
    Implementation of NPSR
    """

    def __init__(self, transform: Union[Transform, None] = None, l=1):
        """
        Constructor

        :param l: peak signal of the image
        """
        super().__init__(transform)
        self.l = l
        self.rmse = RootMeanSquaredError()

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = 20 * torch.log10(self.rmse(x, y) / self.l)
        return out


class StructuralDissimilarity(Metric):
    """
    Implementation of Structural Dissimilarity.
    Computed as (1 - SSIM) / 2
    """

    def __init__(self, transform: Union[Transform, None] = None, window_size=100, k_1=1e-2, k_2=3e-2, l=1):
        """
        Constructor

        :param window_size: number, size of the sliding window in which SSIMs are computed
        :param k_1: component of the first constant (for safe division)
        :param k_2: component of the second constant (for safe division)
        :param l: peak signal, component of the safe division constants
        """
        super().__init__(transform)
        self.window_size = window_size
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("Not an image")
        if x.shape != y.shape:
            raise ValueError("Given images of different shapes")
        
        batch = x.shape[0]
        channels = x.shape[1]
        width = x.shape[2]
        height = x.shape[3]

        num_width_windows = width - self.window_size + 1 if width > self.window_size else 1
        num_height_windows = height - self.window_size + 1 if height > self.window_size else 1

        windows_indexes = [
            [
                range(i_w_start, min(i_w_start + width, i_w_start + self.window_size)),
                range(i_h_start, min(i_h_start + height, i_h_start + self.window_size))
            ]
            for i_w_start, i_h_start in itertools.product(range(num_width_windows), range(num_height_windows))
        ]

        x_windows = torch.zeros(len(windows_indexes), batch, channels, min(self.window_size, width), min(self.window_size, height))
        y_windows = torch.zeros(len(windows_indexes), batch, channels, min(self.window_size, width), min(self.window_size, height))
        for i, indexes in enumerate(windows_indexes):
            x_windows[i, :, :, :, :] += torch.index_select(torch.index_select(x, 2, torch.tensor(indexes[0], dtype=torch.int)), 3, torch.tensor(indexes[1], dtype=torch.int))
            y_windows[i, :, :, :, :] += torch.index_select(torch.index_select(y, 2, torch.tensor(indexes[0], dtype=torch.int)), 3, torch.tensor(indexes[1], dtype=torch.int))
        
        x_means = x_windows.mean(dim=(3, 4))
        y_means = y_windows.mean(dim=(3, 4))
        x_variances = x_windows.var(dim=(3, 4), unbiased=True)
        y_variances = y_windows.var(dim=(3, 4), unbiased=True)
        x_means_expanded = x_means \
            .reshape(num_width_windows * num_height_windows, batch, channels, 1, 1) # \
            # .expand(num_width_windows * num_height_windows, batch, channels, min(self.window_size, width), min(self.window_size, height))
        y_means_expanded = y_means \
            .reshape(num_width_windows * num_height_windows, batch, channels, 1, 1) # \
            # .expand(num_width_windows * num_height_windows, batch, channels, min(self.window_size, width), min(self.window_size, height))
        bessel = (min(self.window_size, width) * min(self.window_size, height))
        bessel = bessel / (bessel - 1)
        # bessel = 1
        cov = bessel * ((x_windows - x_means_expanded) * (y_windows - y_means_expanded)).mean(dim=(3, 4))

        out = (2 * x_means * y_means + self.c_1) * (2 * cov + self.c_2) \
            / ((x_means ** 2 + y_means ** 2 + self.c_1) * (x_variances + y_variances + self.c_2))
        return (1 - out.mean(dim=(0, 2))) / 2
    

class CostMatrixType(Enum):
    L1 = 0
    L2 = 1
    HALF_L2_SQUARED = 2


class WassersteinApproximation(Metric):
    """
    Implemantation of dual-Sinkhorn divergence
    """

    def __init__(
            self,
            transform:Union[Transform, None] = None,
            regularization: float = 5,
            iterations: int = 250,
            division_const: float = 1e-8,
            cost_matrix_type: CostMatrixType = CostMatrixType.L1,
            tolerance: float = 1e-5
        ):
        """
        Constructor
        
        :param transform: transformation made pre-computation
        :param regularization: regularization coefficient of the entropy term in the optimization problem
        :param iterations: fixed number of iterations
        :param division_const: constant added to escape division by zero
        :param cost_matrix_type: type of cost matrics to be used for the computation
        :param tolerance: Possible stopping criterion.
        """
        super().__init__(transform)
        self.regularization = regularization
        self.iterations = iterations
        self.division_const = division_const
        self.cost_matrix_type = cost_matrix_type
        self.tolerance = tolerance

    def compute(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("Not a batch of images")
        if x.shape != y.shape:
            raise ValueError("Given images of different shapes")
        if any(x.flatten() < 0) or any(y.flatten() < 0):
            raise ValueError("Given images are with negative values")
        
        batch = x.shape[0]
        channels = x.shape[1]
        width = x.shape[2]
        height = x.shape[3]

        if channels > 1:
            raise NotImplementedError("Wasserstein not implemented for multi-channel images")
        
        if (x < 0).sum() > 0 or (y < 0).sum() > 0:
            raise ValueError("Images must be given with non-negative entries.")
        
        # Normalization -> into probability distribution
        x_norm, y_norm = (x / x.sum(dim=(2, 3), keepdim=True)).reshape(batch, width * height), \
            (y / y.sum(dim=(2, 3), keepdim=True)).reshape(batch, width * height)
        cost_matrix = None
        if self.cost_matrix_type == CostMatrixType.L1:
            cost_matrix = torch.tensor(
                [
                    [
                        abs(i // width - j // width) + abs(i % width - j % width) 
                        for j in range(width * height)
                    ]
                    for i in range(width * height)
                ]
            )
        elif self.cost_matrix_type == CostMatrixType.L2:
            cost_matrix = torch.tensor(
                [
                    [
                        sqrt(abs(i // width - j // width) ** 2 + abs(i % width - j % width) ** 2) 
                        for j in range(width * height)
                    ]
                    for i in range(width * height)
                ]
            )
        else:
            cost_matrix = torch.tensor(
                [
                    [
                        (1 / 2) * (abs(i // width - j // width) ** 2 + abs(i % width - j % width) ** 2) 
                        for j in range(width * height)
                    ]
                    for i in range(width * height)
                ]
            )

        dists = torch.zeros(batch)

        for i in range(batch):
            dists[i] += self.compute_vectors_distance(x_norm[i].flatten(), y_norm[i].flatten(), cost_matrix)
        return torch.Tensor(dists)
        
    def compute_vectors_distance(self, x, y, cost_matrix):
        indices = (x != 0)
        x_non_zero = x[indices]
        x_non_zero_dim = x_non_zero.shape[0]

        # Algorithm from paper https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf

        # u_vector_prev = torch.ones(x_non_zero_dim) / x_non_zero_dim
        u_vector = torch.ones(x_non_zero_dim) / x_non_zero_dim
        K_matrix = (- self.regularization * cost_matrix[indices, :]).exp()
        K_matrix[K_matrix < self.division_const] = self.division_const
        K_tilde_matrices = torch.diag(1 / x_non_zero) @ K_matrix

        for i in range(self.iterations):
            u_prev = u_vector
            u_vector = 1 / (K_tilde_matrices @ (y / (K_matrix.transpose(0, 1) @ u_vector)))
            if i > 0 and (u_prev - u_vector).norm() < self.tolerance:
                break
        v_vector = y / (K_matrix.transpose(0, 1) @ u_vector)
        
        

        dist = (u_vector * ((K_matrix * cost_matrix[indices, :]) @ v_vector))
        return dist.sum()
