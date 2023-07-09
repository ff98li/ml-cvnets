#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import os
import re
from typing import Any, Dict, List, Optional, Union

from utils import logger
from utils.ddp_utils import is_start_rank_node

from ..layers.conv_layer import Conv2d
from ..layers.linear_layer import LinearLayer
from torch.nn import init

def clean_strip(
    obj: Union[str, List[str]], sep: Optional[str] = ",", strip: bool = True
) -> List[str]:
    # Allowing list of strings as input as well as comma-separated strings
    if isinstance(obj, list):
        strings = obj
    else:
        strings = obj.split(sep)

    if strip:
        strings = [x.strip() for x in strings]
    strings = [x for x in strings if x]
    return strings


def load_pretrained_model(
    model: torch.nn.Module, wt_loc: str, opts: Dict[str, Any], *args, **kwargs
) -> torch.nn.Module:
    """
    Helper function to load pre-trained weights
    """
    if not os.path.isfile(wt_loc):
        logger.error("Pretrained file is not found here: {}".format(wt_loc))

    wts = torch.load(wt_loc, map_location="cpu")

    is_master_node = is_start_rank_node(opts)

    exclude_scopes = getattr(opts, "model.resume_exclude_scopes", "")
    exclude_scopes: List[str] = clean_strip(exclude_scopes)

    missing_scopes = getattr(opts, "model.ignore_missing_scopes", "")
    missing_scopes: List[str] = clean_strip(missing_scopes)

    rename_scopes_map: List[List[str]] = getattr(opts, "model.rename_scopes_map", [])
    if rename_scopes_map:
        for entry in rename_scopes_map:
            if len(entry) != 2:
                raise ValueError(
                    "Every entry in model.rename_scopes_map must contain exactly two string elements"
                    " for before and after. Got {}.".format(str(entry))
                )

    # By default, adding scopes that we exclude to missing scopes
    # If you excluded something, you can't expect it to be there.
    missing_scopes += exclude_scopes

    # remove unwanted scopes
    if exclude_scopes:
        for key in wts.copy():
            if any([re.match(x, key) for x in exclude_scopes]):
                del wts[key]

    if rename_scopes_map:
        for before, after in rename_scopes_map:
            wts = {re.sub(before, after, key): value for key, value in wts.items()}

    strict = not bool(missing_scopes)

    try:
        module = model.module if hasattr(model, "module") else model
        missing_keys, unexpected_keys = module.load_state_dict(wts, strict=strict)

        if unexpected_keys:
            raise Exception(
                "Found unexpected keys: {}."
                "You can ignore these keys using `model.resume_exclude_scopes`.".format(
                    ",".join(unexpected_keys)
                )
            )

        missing_keys = [
            key
            for key in missing_keys
            if not any([re.match(x, key) for x in missing_scopes])
        ]

        if missing_keys:
            raise Exception(
                "Missing keys detected. Did not find the following keys in pre-trained model: {}."
                " You can ignore the keys using `model.ignore_missing_scopes`.".format(
                    ",".join(missing_keys)
                )
            )

        if is_master_node:
            logger.log("Pretrained weights are loaded from {}".format(wt_loc))
    except Exception as e:
        try:
            # Try loading the state dict with size check
            load_state_dict_with_size_check(model, wts)
            if is_master_node:
                logger.log("Pretrained weights are loaded from {}".format(wt_loc))
        except Exception as e:
            if is_master_node:
                logger.error(
                    "Unable to load pretrained weights from {}. Error: {}".format(wt_loc, e)
                )

    return model


def find_and_replace_layer(model, layer_name, new_layer):
    """
    Helper function to replace nested layers
    """
    logger.log(f'Replacing layer: {layer_name}')
    names = layer_name.split(".")
    if len(names) == 1:
        setattr(model, names[0], new_layer)
    elif len(names) == 2:
        layer = getattr(model, names[0])
        setattr(layer, names[1], new_layer)
    else:
        find_and_replace_layer(getattr(model, names[0]), ".".join(names[1:]), new_layer)


def load_state_dict_with_size_check(model, state_dict):
    """
    Load pretrained weights for matched size layers only.
    """
    model_dict = model.state_dict()
    for name, param in state_dict.items():
        # size of the layer's weights doesn't match the size of the corresponding weights in the state_dict
        if name in model_dict and param.size() != model_dict[name].size():
            # check if it's a conv layer
            if 'conv' in name and 'weight' in name:
                # create a new Conv2d layer with the desired size
                new_layer = Conv2d(param.size()[1], param.size()[0], kernel_size=(param.size()[2], param.size()[3]), bias=True)

                # replace the mismatched layer with the new layer
                find_and_replace_layer(model, '.'.join(name.split('.')[:-1]), new_layer)

            elif 'conv' in name and 'bias' in name:
                weight_name = name.replace('bias', 'weight')
                if weight_name in model_dict:
                    # Use the properties of the corresponding weight layer to create a new Conv2d layer
                    weight_size = model_dict[weight_name].size()
                    new_layer = Conv2d(weight_size[1], param.size()[0], kernel_size=(weight_size[2], weight_size[3]), bias=True)
                    init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')

                    # Replace the mismatched layer with the new layer
                    find_and_replace_layer(model, '.'.join(name.split('.')[:-1]), new_layer)

    # Load the state dict
    model.load_state_dict(state_dict)


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs
):
    module_name = kwargs.get("module_name", "")
    with_decay = []
    without_decay = []
    with_decay_param_names = []
    without_decay_param_names = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                    without_decay_param_names.append(module_name + p_name)
                elif param.requires_grad:
                    with_decay.append(param)
                    with_decay_param_names.append(module_name + p_name)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
                without_decay_param_names.append(module_name + p_name)
            elif param.requires_grad:
                with_decay.append(param)
                with_decay_param_names.append(module_name + p_name)
    param_list = [
        {
            "params": with_decay,
            "weight_decay": weight_decay,
            "param_names": with_decay_param_names,
        }
    ]
    if len(without_decay) > 0:
        param_list.append(
            {
                "params": without_decay,
                "weight_decay": 0.0,
                "param_names": without_decay_param_names,
            }
        )
    return param_list
