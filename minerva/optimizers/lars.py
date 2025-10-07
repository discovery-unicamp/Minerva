import torch
from torch.optim.optimizer import Optimizer
from typing import Set, Optional, Callable, Any


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    This optimizer implements the LARS algorithm, which adapts the learning rate
    for each layer based on the ratio of the weight norm to the gradient norm.
    This helps stabilize training and allows for larger learning rates.
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        eta: float = 0.001,
        epsilon: float = 1e-8,
        exclude_from_layer_adaptation: Optional[Set[str]] = None,
    ):
        """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

        This optimizer implements the LARS algorithm, which adapts the learning rate
        for each layer based on the ratio of the weight norm to the gradient norm.
        This helps stabilize training and allows for larger learning rates.

        Parameters
        ----------
        params : Any
            Parameters to optimize.
        lr : float
            Base learning rate.
        momentum : float, optional, default: 0.9
            Momentum factor.
        weight_decay : float, optional, default: 1e-6
            Weight decay (L2 penalty) coefficient.
        eta : float, optional, default: 0.001
            Trust coefficient for layer-wise rate scaling.
        epsilon : float, optional, default: 1e-8
            Small constant for numerical stability.
        exclude_from_layer_adaptation : Set[str], optional
            Set of parameter names to exclude from layer-wise adaptation
            (e.g., batch normalization layers and biases).

        Attributes
        ----------
        exclude_set : Set[str]
            Set of parameter names excluded from layer-wise adaptation.

        References
        ----------
        .. [1] You, Yang, et al. "Large batch training of convolutional networks."
            arXiv preprint arXiv:1708.03888 (2017).

        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            epsilon=epsilon,
        )
        super(LARS, self).__init__(params, defaults)
        self.exclude_set = exclude_from_layer_adaptation or set()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss : torch.Tensor or None
            Loss from the closure if provided, otherwise None.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Extract hyperparameters
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                eta = group["eta"]
                epsilon = group["epsilon"]
                grad = p.grad.data

                # Get state
                state = self.state[p]

                # Initialize momentum buffer if needed
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                # Get parameter name for exclusion check
                param_name = getattr(p, "param_name", "")

                # ===== LARS core calculation =====
                if param_name in self.exclude_set:
                    # For excluded parameters (BN, bias), use standard LR
                    trust_ratio = 1.0
                else:
                    # Compute weight norm and raw gradient norm
                    w_norm = torch.norm(p.data)
                    g_norm = torch.norm(grad)

                    # Compute trust ratio (local learning rate scaling)
                    denom = g_norm + weight_decay * w_norm + epsilon
                    trust_ratio = (
                        eta * w_norm / denom if w_norm > 0 and denom > 0 else 1.0
                    )

                # Calculate effective learning rate
                effective_lr = lr * trust_ratio

                # Apply weight decay to gradient
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update momentum buffer
                state["momentum_buffer"].mul_(momentum).add_(grad, alpha=effective_lr)

                # Update weights
                p.data.sub_(state["momentum_buffer"])

        return loss
