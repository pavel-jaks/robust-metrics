from typing import Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Transform(nn.Module):
    pass


class Identity(Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Metric(nn.Module):
    def __init__(self, transform: Union[Transform, None] = None):
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
        pass


class Norm(nn.Module):
    pass


class LpNorm(Norm):
    def __init__(self, p: int):
        super().__init__()
        if p < 1:
            raise ValueError("p must be greater than 0")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x ** self.p).sum(dim=tuple(range(1, x.ndim))) ** (1 / self.p)


class L2Norm(LpNorm):
    def __init__(self):
        super().__init__(2)


class L1Norm(LpNorm):
    def __init__(self):
        super().__init__(1)


class LinfNorm(Norm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.abs()
        for _ in range(1, out.ndim):
            out = out.max(dim=1)[0]
        return out


class L0Norm(Norm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x != torch.zeros(x.shape)) * 1).sum(dim=tuple(range(1, x.ndim)))


class MetricFromNorm(Metric):
    def __init__(self, norm: Norm, transform: Union[Transform, None] = None):
        super().__init__(transform)
        self.norm = norm

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # print(f'{self.norm}: {self.norm(x - y)}')
        out = self.norm(x - y)
        return out


class LpMetric(MetricFromNorm):
    def __init__(self, p, transform: Union[Transform, None] = None):
        super().__init__(LpNorm(p), transform)


class L2Metric(LpMetric):
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
