#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Callable, Optional, Sequence, Union

from torch import nn
from transformers.activations import get_activation


def MLP(
    layer_sizes: Sequence[int],
    dropout: Optional[float] = None,
    activation: Optional[Union[str, Callable]] = None,
    output_dropout: Optional[Union[float, bool]] = None,
    output_activation: Optional[Union[str, bool, Callable]] = None,
):
    layers = []
    num_layers = len(layer_sizes) - 1
    for index in range(num_layers):
        if index < num_layers - 1:
            layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))

            if isinstance(activation, str):
                layers.append(get_activation(activation))
            elif isinstance(activation, Callable):
                layers.append(activation())

            if isinstance(dropout, float):
                layers.append(nn.Dropout(dropout))
        else:
            layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))

            if isinstance(output_activation, str):
                layers.append(get_activation(output_activation))
            elif isinstance(output_activation, Callable):
                layers.append(output_activation())
            elif output_activation is True and activation is not None:
                if isinstance(activation, str):
                    layers.append(get_activation(activation))
                elif isinstance(activation, Callable):
                    layers.append(activation())

            if isinstance(output_dropout, float):
                layers.append(nn.Dropout(p=output_dropout))
            elif output_dropout is True and isinstance(dropout, float):
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def main():
    mlp = MLP([768, 768, 128])
    print(mlp)


if __name__ == "__main__":
    main()
