import json
from functools import partial
from torch import optim as optim

def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def build_finetune_optimizer(model, depths, opt_params):
    opt_type = opt_params['opt_type']
    base_lr = opt_params['base_lr']
    layer_decay = opt_params['layer_decay']
    weight_decay = opt_params['weight_decay']

    depths = depths
    num_layers = sum(depths)
    get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
    
    scales = list(layer_decay ** i for i in reversed(range(num_layers + 2)))
    
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_finetune_param_groups(
        model, base_lr, weight_decay,
        get_layer_func, scales, skip, skip_keywords)
    
    optimizer = None
    if opt_type == 'sgd':
        optimizer = optim.SGD(parameters, momentum=opt_params['momentume'], nesterov=True,
                              lr=base_lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        
        optimizer = optim.AdamW(parameters, eps=opt_params['eps'], betas=opt_params['betas'],
                                lr=base_lr, weight_decay=weight_decay)

    return optimizer
