import re


def get_layer_lrs_with_crf(
    named_parameters,
    transformer_prefix,
    learning_rate,
    layer_decay,
    n_layers,
    crf_prefix="crf",
    crf_ratio=10.0,
):
    groups = []
    crf_groups = []
    temp_groups = [None] * (n_layers + 3)
    temp_no_decay_groups = [None] * (n_layers + 3)
    regex = rf"^{transformer_prefix}\.(embeddings|encoder)\w*\.(layer.(\d+))?.+"
    regex = re.compile(regex)
    for name, parameters in named_parameters:
        m = regex.match(name)

        is_transformer = True
        if m is None:
            depth = n_layers + 2
            is_transformer = False
        elif m.group(1) == "embeddings":
            depth = 0
        elif m.group(1) == "encoder":
            depth = int(m.group(3)) + 1
        else:
            raise Exception("Not Recommend!!!")

        if is_transformer and any(
            x in name for x in ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        ):
            if temp_no_decay_groups[depth] is None:
                temp_no_decay_groups[depth] = []
            temp_no_decay_groups[depth].append(parameters)
        elif not is_transformer and crf_prefix in name:
            crf_groups.append(parameters)
        else:
            if temp_groups[depth] is None:
                temp_groups[depth] = []
            temp_groups[depth].append(parameters)

    for depth, parameters in enumerate(temp_no_decay_groups):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "weight_decay": 0.0,
                    "lr": learning_rate * (layer_decay ** (n_layers + 2 - depth)),
                }
            )
    for depth, parameters in enumerate(temp_groups):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate * (layer_decay ** (n_layers + 2 - depth)),
                }
            )
    if crf_groups:
        groups.append({"params": crf_groups, "lr": learning_rate * crf_ratio})

    return groups
