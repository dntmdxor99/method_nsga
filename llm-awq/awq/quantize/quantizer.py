# from turtle import up
import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    assert n_bit == int(n_bit), "n_bit should be integer"
    assert q_group_size != 0, "q_group_size should not be 0"

    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
    

@torch.no_grad()
def pseudo_quantize_model_weight_bit_adjust(
    model,
    w_bit,
    q_config,

    ## customizing
    arch = None,
    owq = None,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization bit adjust per linear..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            if int(arch[n][i]) != arch[n][i]:
                if owq is not None:
                    original = {}
                    for linear in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
                        module, proj = linear.split('.')
                        key = f'model.layers.{i}.{linear}'
                        original[key] = getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]].clone()

                        # getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]].data.zero_()
                        getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]] = 0

                m.cuda()
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=int(arch[n][i]),
                    zero_point = q_config["zero_point"], q_group_size = 64 if int(arch[n][i]) == 2 else 128
                )
                m.cpu()

                if owq is not None:
                    for linear in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
                        module, proj = linear.split('.')
                        key = f'model.layers.{i}.{linear}'
                        # getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]].data.copy_(original[key])
                        getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]] = original[key]

                        del original[key]
            elif int(arch[n][i]) == arch[n][i]:
                m.cuda()
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=int(arch[n][i]),
                    zero_point = q_config["zero_point"], q_group_size = 64 if int(arch[n][i]) == 2 else 128
                )
                m.cpu()
            else:
                raise NotImplementedError(f"Unsupported bit adjust value {arch[n][i]}")



@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,

    ## customizing
    bit_adjust_per_linear = None,
    arch = None,
    owq = None,
):
    if bit_adjust_per_linear:
        print('bit_adjust_per_linear')
        pseudo_quantize_model_weight_bit_adjust(model, w_bit, q_config, arch = arch, owq = owq)
        return
    from .pre_quant import get_blocks, get_named_linears

    assert owq is None, "owq is not supported in this function"

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def pseudo_quantize_model_weight_wBit_setting(
    model,
    w_bit,
    q_config,
    archs = None,
    search_space_config = None,
    flag = None
):
    from .pre_quant import get_blocks, get_named_linears
    from copy import deepcopy
    # from .utils.data_utils import get_loader
    # from .utils.func_utils import get_net_info
    # from .utils.eval_utils import eval_metric
    from utils.data_utils import get_loader
    from utils.func_utils import get_net_info
    from utils.eval_utils import eval_metric

    import json
    import time
    import os

    print(len(archs))

    original_model = model.to('cpu')
    original_model.seqlen = 2048

    # datasets = ['wikitext2']
    # train_loaders = {dataset: get_loader(dataset, model='/SSD/Woo/llama-2-7B-hf', n_sample=128 , train=True, seed=0, seqlen=original_model.seqlen) for dataset in datasets}
    # test_loaders = {dataset: get_loader(dataset, model='/SSD/Woo/llama-2-7B-hf', train=False, seqlen=original_model.seqlen) for dataset in datasets}

    # for dataset in ['wikitext2']:
    #     if not os.path.exists('algorithm_awq'):
    #         os.makedirs('algorithm_awq')
    #     if not os.path.exists(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_ppl_wBits_{flag}.json'):
    #         with open(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_ppl_wBits_{flag}.json', 'w') as f:
    #             json.dump({'archive': []}, f, ensure_ascii=False, indent=4)
    #     if not os.path.exists(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_loss_wBits_{flag}.json'):
    #         with open(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_loss_wBits_{flag}.json', 'w') as f:
    #             json.dump({'archive': []}, f, ensure_ascii=False, indent=4)

    for arch in archs:
        model = deepcopy(original_model)
        layers = get_blocks(model)
        for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
            q_w_bit = arch['self_attn.q_proj'][i]
            q_group_size = 64 if q_w_bit == 2 else 128
            k_w_bit = arch['self_attn.k_proj'][i]
            k_group_size = 64 if k_w_bit == 2 else 128
            v_w_bit = arch['self_attn.v_proj'][i]
            v_group_size = 64 if v_w_bit == 2 else 128
            o_w_bit = arch['self_attn.o_proj'][i]
            o_group_size = 64 if o_w_bit == 2 else 128
            up_w_bit = arch['mlp.up_proj'][i]
            up_group_size = 64 if up_w_bit == 2 else 128
            gate_w_bit = arch['mlp.gate_proj'][i]
            gate_group_size = 64 if gate_w_bit == 2 else 128
            down_w_bit = arch['mlp.down_proj'][i]
            down_group_size = 64 if down_w_bit == 2 else 128
            
            config = {'self_attn.q_proj':[q_w_bit, q_group_size], 'self_attn.k_proj':[k_w_bit, k_group_size], 'self_attn.v_proj':[v_w_bit, v_group_size], 'self_attn.o_proj':[o_w_bit, o_group_size], 
                      'mlp.up_proj':[up_w_bit, up_group_size], 'mlp.gate_proj':[gate_w_bit, gate_group_size], 'mlp.down_proj':[down_w_bit, down_group_size]}

            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                m.cuda()

                w_bit = config[n][0]
                q_config["q_group_size"] = config[n][1]
                # print(f'layer : {i}.{n}, w_bit : {w_bit}, group_size : {q_config["q_group_size"]}')

                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=w_bit, **q_config
                )
                m.cpu()

        import code; code.interact("quantizer, line 203", local=locals())
        
        model = model.to('cuda')

        # for metric in ['ppl', 'loss']:
        #     if metric == 'ppl':
        #         loaders = test_loaders
        #     elif metric == 'loss':
        #         loaders = train_loaders
        #     else:
        #         NotImplementedError(f'metric should be ppl or loss, but got {metric}')

        #     metric_list = dict()
        #     for dataset, loader in loaders.items():
        #         metric_list[dataset] = eval_metric(model=model, metric=metric, loader=loader, device='cuda:0', seqlen=model.seqlen)
        #         complexity = get_net_info(arch, search_space_config)

        #         archive = [arch, metric_list[dataset], complexity['bits']]

        #         print(f'{dataset} {metric} : {metric_list[dataset]}')
        #         print(f'complexity : {complexity["bits"]}')

        #         for _ in range(5):
        #             try:
        #                 if metric == 'ppl':
        #                     with open(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_ppl_wBits_{flag}.json', 'r') as f:
        #                         data = json.load(f)
        #                         data['archive'].append(archive)
        #                         with open(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_ppl_wBits_{flag}.json', 'w') as f:
        #                             # json.dump({'archive': archive}, f, ensure_ascii=False, indent=4)
        #                             json.dump(data, f, ensure_ascii=False, indent=4)
        #                 elif metric == 'loss':
        #                     with open(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_loss_wBits_{flag}.json', 'r') as f:
        #                         data = json.load(f)
        #                         data['archive'].append(archive)
        #                         with open(f'algorithm_awq/algorithm_llama-2-7B-hf_{dataset}_loss_wBits_{flag}.json', 'w') as f:
        #                             # json.dump({'archive': archive}, f, ensure_ascii=False, indent=4)
        #                             json.dump(data, f, ensure_ascii=False, indent=4)
        #                 break
        #             except:
        #                 time.sleep(5)
        #         else:
        #             continue

        del model
        torch.cuda.empty_cache()
        gc.collect()
        


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
