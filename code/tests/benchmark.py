import timeit

import torch

import metrics


metric_wasserstein = metrics.WassersteinApproximation(regularization=3)
metric_dssim5 = metrics.StructuralDissimilarity(window_size=5)
metric_dssim10 = metrics.StructuralDissimilarity(window_size=10)
metric_dssim20 = metrics.StructuralDissimilarity(window_size=20)
metric_dssim100 = metrics.StructuralDissimilarity()
metric_l2 = metrics.L2Metric()
metric_l1 = metrics.LpMetric(1)
metric_l0 = metrics.L0Metric()
metric_linf = metrics.LinfMetric()
metric_npsr = metrics.NoiseToPeakSignalRatio()


def benchmark(metric, image_pair, number_of_runs, description):
    setup = f"from __main__ import {metric}; import torch; image1 = {image_pair[0]}; image2 = {image_pair[1]};"
    code = f"{metric}(image1, image2)"
    time = timeit.repeat(stmt=code, setup=setup, repeat=10, number=number_of_runs)
    print(description)
    print(f"Metric: {metric}")
    for i, result in enumerate(time):
        print(f"Run {i}: {result}")
    print("_____________")


def main():
    image_pairs = [
        ('torch.rand(10, 1, 28, 28)', 'torch.rand(10, 1, 28, 28)'),
        ('torch.rand(100, 1, 28, 28)', 'torch.rand(100, 1, 28, 28)'),
        ('torch.rand(1000, 1, 28, 28)', 'torch.rand(1000, 1, 28, 28)'),
        ('torch.rand(10000, 1, 28, 28)', 'torch.rand(10000, 1, 28, 28)')
    ]
    benchmark(f'{metric_l2=}'.split('=')[0], image_pairs[0], 100, "test")



if __name__ == '__main__':
    main()
