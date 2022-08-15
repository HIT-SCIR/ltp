from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from transformers.file_utils import ExplicitEnum
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    EXPONENTIAL = "exponential"


def get_exponential_schedule(optimizer: Optimizer, gamma: float, last_epoch: int = -1):
    return ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.EXPONENTIAL: get_exponential_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    gamma: float = 0.5,
    num_cycles: Optional[Union[int, float]] = None,
    lr_end: float = 1e-7,
    power: float = 1.0,
):
    """Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if name == SchedulerType.EXPONENTIAL:
        return schedule_func(optimizer, gamma=gamma)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE:
        num_cycles = 0.5 if num_cycles is None else num_cycles
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        num_cycles = 1 if num_cycles is None else num_cycles
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=lr_end,
            power=power,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def compose_with_scheduler(
    optimizer,
    num_training_steps,
    scheduler_type,
    scheduler_args=None,
    warmup_ratio=None,
    interval="step",
    frequency="1",
):
    if isinstance(warmup_ratio, float):
        num_warmup_steps = int(warmup_ratio * num_training_steps)
    elif isinstance(warmup_ratio, int):
        num_warmup_steps = warmup_ratio
    else:
        num_warmup_steps = None

    if scheduler_args is None:
        scheduler_args = {}

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **scheduler_args,
            ),
            "interval": interval,
            "frequency": frequency,
        },
    }
