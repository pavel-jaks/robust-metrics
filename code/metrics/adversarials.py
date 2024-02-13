import torch
import torch.nn as nn

from metrics import Metric

def cw_batch(
        model: nn.Module,
        benign_examples: torch.Tensor,
        labels: torch.Tensor,
        c_lambda: float,
        metric: Metric,
        special_init: bool = False
    ) -> torch.Tensor:
    if special_init:
        adversarial_examples = benign_examples
    else:
        adversarial_examples = 0.5 * torch.ones(benign_examples.shape) \
            + 0.3 * (2 * torch.rand(benign_examples.shape) - 1)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    step_size = 1e-2
    for _ in range(100):
        adversarial_examples.requires_grad = True
        if adversarial_examples.grad is not None:
            adversarial_examples.grad.zero_()
        benign_examples.requires_grad = True
        if benign_examples.grad is not None:
            benign_examples.grad.zero_()
        metrics = metric(benign_examples, adversarial_examples)
        loss = metrics.sum() \
            - c_lambda * loss_fn(
                model(adversarial_examples),
                torch.tensor(labels, dtype=torch.long)
        )
        loss.backward()
        adversarial_examples = (adversarial_examples \
            - step_size * adversarial_examples.grad.apply_(
                lambda x: 1 if x >= 0 else -1
            )).detach()
    return adversarial_examples
