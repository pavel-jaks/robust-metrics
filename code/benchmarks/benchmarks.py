import asv
import torch

import metrics
import tests


class MetricsBenchmarkClass:
    params = [
        (torch.rand(10, 1, 5, 5), torch.rand(10, 1, 5, 5)),
        (torch.rand(10, 1, 10, 10), torch.rand(10, 1, 10, 10)),
        (torch.rand(10, 1, 15, 15), torch.rand(10, 1, 15, 15)),
        (torch.rand(10, 1, 20, 20), torch.rand(10, 1, 20, 20)),
        (torch.rand(10, 1, 25, 25), torch.rand(10, 1, 25, 25)),
        (torch.rand(10, 1, 30, 30), torch.rand(10, 1, 30, 30))
    ]

    def time_L2MetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.L2Metric()
        metric(image1, image2)

    def time_L1MetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.LpMetric(1)
        metric(image1, image2)
    
    def time_LInfMetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.LinfMetric()
        metric(image1, image2)
    
    def time_L0MetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.L0Metric()
        metric(image1, image2)

    def time_MSEMetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.MeanSquaredError()
        metric(image1, image2)

    def time_RMSEMetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.RootMeanSquaredError()
        metric(image1, image2)

    def time_PSNRMetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.PeakSignalToNoiseRatio()
        metric(image1, image2)

    def time_DSSIMMetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.StructuralDissimilarity()
        metric(image1, image2)
    
    def time_WassersteinMetricBenchmark(self, images):
        image1, image2 = images
        metric = metrics.WassersteinApproximation(cost_matrix_type=metrics.CostMatrixType.L2)
        metric(image1, image2)
