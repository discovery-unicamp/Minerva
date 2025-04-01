from typing import Any, Callable, Dict, Optional, overload

import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    """Implements the Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    Implementation borrowed from lightly SSL library.
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0.9,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        """Constructs a new LARS optimizer.

        Parameters
        ----------
        params : Any
            Parameters to optimize.
        lr : float
            Learning rate.
        momentum : float, optional
            Momentum factor, by default 0.9
        dampening : float, optional
            Dampening for momentum, by default 0
        weight_decay : float, optional
            Weight decay (L2 penalty), by default 0.9
        nesterov : bool, optional
            Enables Nesterov momentum, by default False
        trust_coefficient : float, optional
            Trust coefficient for computing learning rate, by default 0.001
        eps : float, optional
            Eps for division denominator, by default 1e-8

        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    # Type ignore for overloads is required for Python 3.7.
    @overload  # type: ignore[override]
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Exclude scaling for params with 0 weight decay.
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # Apply Lars scaling and weight decay.
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["trust_coefficient"]

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # Apply momentum.
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
