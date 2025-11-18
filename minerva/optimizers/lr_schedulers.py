from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_iter: int,
        power: float = 0.9,
        min_lr: float = 1e-4,
        last_epoch: int = -1,
    ):
        """
        Polynomial decay LR scheduler.

        The learning rate decays from the initial ``base_lr`` to at least ``min_lr``
        following a polynomial schedule:

        :math:`lr = \\max(\\text{min\\_lr}, \\text{base\\_lr} \\cdot (1 - \\tfrac{t}{T})^{power})`

        where :math:`t` is the current step (``last_epoch``) and :math:`T` is ``max_iter``.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Wrapped optimizer.
        max_iter : int
            Total number of iterations (epochs or steps, depending on usage).
            Must be strictly greater than 0. When ``last_epoch`` reaches or exceeds
            ``max_iter``, the learning rate is clamped to ``min_lr``.
        power : float, default=0.9
            Power factor for polynomial decay, controlling how fast the learning rate decays.
        min_lr : float, default=1e-4
            Minimum learning rate allowed. Once the polynomial decay falls below this
            value, the learning rate is fixed at ``min_lr``.
        last_epoch : int, default=-1
            The index of the last epoch. If set to ``-1`` (default), the scheduler
            initializes with the optimizer’s learning rates, and the first call to
            ``step()`` sets the learning rate to ``base_lr`` without applying decay.
        """
        self.max_iter = max_iter
        self.power = power
        self.min_lr = min_lr
        assert max_iter > 0, "max_iter must be greater than 0"
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Clamp last_epoch to be at most max_iter to avoid negative/zero/complex factors
        epoch = min(self.last_epoch, self.max_iter)
        factor = max(0, 1 - epoch / self.max_iter) ** self.power
        # Ensure LR never goes below min_lr
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]
