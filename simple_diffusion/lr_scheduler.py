import math


class CosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        decay_iters: int,
        min_lr: float,
        lr: float,
        verbose: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.min_lr = min_lr
        self.lr = lr
        self.verbose = verbose
        self.iter = 1

    def step(self) -> None:
        if self.iter <= self.warmup_iters:
            lr = self.lr * self.iter / self.warmup_iters
        elif self.iter >= self.decay_iters:
            lr = self.min_lr
        else:
            decay = (self.iter - self.warmup_iters) / (
                self.decay_iters - self.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
            lr = self.min_lr + coeff * (self.lr - self.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.verbose:
            print(self.iter, lr)
        self.iter += 1
