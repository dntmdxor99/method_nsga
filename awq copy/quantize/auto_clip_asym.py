import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc

__all__ = ["auto_clip_block"]

@torch.no_grad()
def auto_clip_block_asym(module, w_bit, q_config, input_feat, bit_diff_while_smoothing = False, module_bit = None):
    if bit_diff_while_smoothing:
        return auto_clip_block_bit_adjust(module, w_bit, q_config, input_feat, bit_diff_while_smoothing, module_bit)
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    # clip_list = []
    max_clip_list = []
    min_clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val, min_val = auto_clip_layer_asym(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        # clip_list.append((name, max_val))
        max_clip_list.append((name, max_val))
        min_clip_list.append((name, min_val))
        named_linears[name].cpu()
    # return clip_list
    return max_clip_list, min_clip_list

@torch.no_grad()
def auto_clip_layer_asym(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    best_min_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        # org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        org_max_val = w.amax(dim=-1, keepdim=True)
        org_min_val = w.amin(dim=-1, keepdim=True)

        best_max_val = org_max_val.clone()
        best_min_val = org_min_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        org_out_dict = {}

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = org_min_val * (1 - i_s / n_grid)
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_min_val[cur_best_idx] = min_val[cur_best_idx]
        best_max_val_all.append(best_max_val)
        best_min_val_all.append(best_min_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_min_val = torch.cat(best_min_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1), best_min_val.squeeze(1)


@torch.no_grad()
def auto_clip_block_asym_bit_adjust(module, w_bit, q_config, input_feat, module_bit = None):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    max_clip_list = []
    min_clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue      
        named_linears[name].cuda()
        q_config['q_group_size'] = 64 if module_bit[name] == 2 else 128
        max_val, min_val = auto_clip_layer_asym(
            # named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
            named_linears[name].weight, input_feat[name], n_bit=module_bit[name], q_config=q_config
        )
        max_clip_list.append((name, max_val))
        min_clip_list.append((name, min_val))
        named_linears[name].cpu()
    return max_clip_list, min_clip_list


@torch.no_grad()
def auto_clip_block_asym(module, w_bit, q_config, input_feat, bit_adjust = False, module_bit = None):
    if bit_adjust:
        return auto_clip_block_asym_bit_adjust(module, w_bit, q_config, input_feat, module_bit)
    
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    max_clip_list = []
    min_clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val, min_val = auto_clip_layer_asym(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        max_clip_list.append((name, max_val))
        min_clip_list.append((name, min_val))
        named_linears[name].cpu()
    return max_clip_list, min_clip_list


@torch.no_grad()
def apply_clip_asym(module, max_clip, min_clip):
    from ..utils.module import get_op_by_name

    for name, max_val in max_clip:
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, max = max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()

    for name, min_val in min_clip:
        layer = get_op_by_name(module, name)
        layer.cuda()
        min_val = min_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*min_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, min = min_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()