# This code was copied from the DiffusionTS implementation in
# https://github.com/Y-debug-sys/Diffusion-TS/blob/main/engine/lr_sch.py
from torch import inf
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateauWithWarmup(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        warmup_lr: float or None, the learning rate to be touched after warmup
        warmup: int, the number of steps to warmup
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        warmup_lr=None,
        warmup=0,
    ):
        """Configure linear warmup and metric-based learning-rate reductions."""
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.warmup_lr = warmup_lr
        self.warmup = warmup

        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _prepare_for_warmup(self):
        """Compute per-group warmup increments from the initial and target rates."""
        if self.warmup_lr is not None:
            if isinstance(self.warmup_lr, (list, tuple)):
                if len(self.warmup_lr) != len(self.optimizer.param_groups):
                    raise ValueError(
                        "expected {} warmup_lrs, got {}".format(
                            len(self.optimizer.param_groups), len(self.warmup_lr)
                        )
                    )
                self.warmup_lrs = list(self.warmup_lr)
            else:
                self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
        else:
            self.warmup_lrs = None
        if self.warmup > self.last_epoch:
            curr_lrs = [group["lr"] for group in self.optimizer.param_groups]
            self.warmup_lr_steps = [
                max(0, (self.warmup_lrs[i] - curr_lrs[i]) / float(self.warmup))
                for i in range(len(curr_lrs))
            ]
        else:
            self.warmup_lr_steps = None

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        """Advance warmup or update plateau tracking using the supplied metric."""
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup:
            self._increase_lr(epoch)
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        """Reduce each learning rate without crossing its configured minimum."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(
                        "Epoch {:5d}: reducing learning rate"
                        " of group {} to {:.4e}.".format(epoch, i, new_lr)
                    )

    def _increase_lr(self, epoch):
        # used for warmup
        """Increase learning rates by their warmup increments."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr + self.warmup_lr_steps[i], self.min_lrs[i])
            param_group["lr"] = new_lr
            if self.verbose:
                print(
                    "Epoch {:5d}: increasing learning rate"
                    " of group {} to {:.4e}.".format(epoch, i, new_lr)
                )

    @property
    def in_cooldown(self):
        """Return whether plateau reductions are temporarily suspended."""
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        """Compare a metric with the best value using the configured threshold."""
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        """Validate comparison settings and initialize the worst possible metric."""
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self._prepare_for_warmup()

    def state_dict(self):
        """Return scheduler state without the optimizer reference."""
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Restore scheduler state and its metric comparison settings."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
