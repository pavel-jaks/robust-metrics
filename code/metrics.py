from typing import Union
from abc import abstractmethod

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

    pass


class LpNorm(Norm):
    """
    Implementation of a L_p norm for a given positive integer p
    """

    def __init__(self, p: int):
        """
        :param p: positive integer
        """
        super().__init__()
        if p < 1:
            raise ValueError("p must be greater than 0")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of which norm shall be computed (shape: Batch x the_rest)

        :return: Tensor of batch of norms
        """
        return (x ** self.p).sum(dim=tuple(range(1, x.ndim))) ** (1 / self.p)


class L2Norm(LpNorm):
    """
    Special case of a LpNorm - L_2
    """
    
    def __init__(self):
        super().__init__(2)


class L1Norm(LpNorm):
    """
    Special case of a LpNorm - L_1
    """
    def __init__(self):
        super().__init__(1)


class LinfNorm(Norm):
    """
    Implementation of a L_infty norm
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
    
    def __init__(self, p, transform: Union[Transform, None] = None):
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
    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(L0Norm(), transform)


class LinfMetric(MetricFromNorm):
    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(LinfNorm(), transform)


class MeanSquaredError(Metric):
    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(transform)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((x - y) ** 2).mean(dim=tuple(range(1, x.ndim)))


class RootMeanSquaredError(Metric):
    def __init__(self, transform: Union[Transform, None] = None):
        super().__init__(transform)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((x - y) ** 2).mean(dim=tuple(range(1, x.ndim))) ** (1 / 2)
