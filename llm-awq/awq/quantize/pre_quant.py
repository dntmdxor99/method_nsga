import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tinychat.models import LlavaLlamaForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip
from .auto_clip_asym import auto_clip_block_asym, apply_clip_asym

from .auto_scale_bit_adjust_per_linear import auto_scale_block_bit_adjust_per_linear
from .auto_scale_bit_adjust_per_linear_owq import auto_scale_block_bit_adjust_per_linear_owq

__all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, LlavaLlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq_bit_adjust(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval",

    ## customizing
    arch = None,
    bit_adjust_per_linear = False,
    clip_asym = False,
    owq = None,
):
    assert arch is not None
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    if clip_asym:
        awq_results = {
            "scale": [],
            "min_clip": [],
            "max_clip": [],
        }
    else:
        awq_results = {
            "scale": [],
            "clip": [],
        }

    if owq is not None:
        auto_scale_func = auto_scale_block_bit_adjust_per_linear_owq
        print('apply owq')
    else:
        auto_scale_func = auto_scale_block_bit_adjust_per_linear
    

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_func(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,

                ## customizing
                module_bit = {proj : arch[proj][i] for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']},
                owq_layer = {proj : owq[f'model.layers.{i}.{proj}'] for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']} if owq is not None else None,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            if clip_asym:
                max_clip_list, min_clip_list = auto_clip_block_asym(
                    layer,
                    w_bit=w_bit,
                    q_config=q_config,
                    input_feat=input_feat,

                    ## customizing
                    bit_adjust = True,
                    module_bit = {proj : int(arch[proj][i]) for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']},
                    # owq_layer = {proj : owq[f'model.layers.{i}.{proj}'] for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']} if owq is not None else None,
                )
                apply_clip_asym(layer, max_clip_list, min_clip_list)
                awq_results['max_clip'] += append_str_prefix(
                    max_clip_list, get_op_name(model, layer) + "."
                )
                awq_results['min_clip'] += append_str_prefix(
                    min_clip_list, get_op_name(model, layer) + "."
                )
            else:
                print('clip')
                clip_list = auto_clip_block(
                    layer,
                    w_bit=w_bit,
                    q_config=q_config,
                    input_feat=input_feat,

                    ## customizing
                    bit_adjust = True,
                    module_bit = {proj : int(arch[proj][i]) for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']},
                    # owq_layer = {proj : owq[f'model.layers.{i}.{proj}'] for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']} if owq is not None else None,

                )
                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(
                    clip_list, get_op_name(model, layer) + "."
                )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval",

    ## customizing
    quantflow = False,
    bit_adjust_per_linear = False,
    arch = None,
    clip_asym = False,
    owq = None,
):
    if quantflow:
        return run_awq_quantflow(model, enc, w_bit, q_config, n_samples, seqlen, auto_scale, mse_range, calib_data, clip_asym = clip_asym, owq = owq)
    if bit_adjust_per_linear:
        print('bit_adjust_per_linear')
        print("clip_asym", clip_asym)
        return run_awq_bit_adjust(model, enc, w_bit, q_config, n_samples, seqlen, auto_scale, mse_range, calib_data, arch, 
                                  bit_adjust_per_linear = bit_adjust_per_linear, clip_asym = clip_asym, owq = owq)

    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    if clip_asym:
        awq_results = {
            "scale": [],
            "min_clip": [],
            "max_clip": [],
        }
    else:
        awq_results = {
            "scale": [],
            "clip": [],
        }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            if clip_asym:
                max_clip_list, min_clip_list = auto_clip_block_asym(
                    layer,
                    w_bit=w_bit,
                    q_config=q_config,
                    input_feat=input_feat,
                )
                apply_clip_asym(layer, max_clip_list, min_clip_list)
                awq_results['max_clip'] += append_str_prefix(
                    max_clip_list, get_op_name(model, layer) + "."
                )
                awq_results['min_clip'] += append_str_prefix(
                    min_clip_list, get_op_name(model, layer) + "."
                )
            else:
                clip_list = auto_clip_block(
                    layer,
                    w_bit=w_bit,
                    q_config=q_config,
                    input_feat=input_feat,
                )
                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(
                    clip_list, get_op_name(model, layer) + "."
                )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


@torch.no_grad()
def apply_awq(model, awq_results, owq = None):
    # if owq is not None:
    #     original = {}
    #     # keys() of owq is model.layers.n.self_attn.q_proj, etc.
    #     # values() of owq is column list
    #     for i in range(len(model.model.layers)):
    #         for linear in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
    #             module, proj = linear.split('.')
    #             key = f'model.layers.{i}.{linear}'
    #             original[key] = getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]].clone()

    #             # getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]].data.zero_()
    #             getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]] = 0
                
    apply_scale(model, awq_results["scale"])
    if "clip" in awq_results:
        apply_clip(model, awq_results["clip"])
    elif "max_clip" in awq_results:
        apply_clip_asym(model, awq_results["max_clip"], awq_results["min_clip"])

    # if owq is not None:
    #     for i in range(len(model.model.layers)):
    #         for linear in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
    #             module, proj = linear.split('.')
    #             key = f'model.layers.{i}.{linear}'
    #             # getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]].data.copy_(original[key])
    #             getattr(getattr(model.model.layers[i], module), proj).weight[:, owq[key]] = original[key]

    #             del original[key]


## customizing for true quantflow
@torch.no_grad()
def run_awq_quantflow(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
):
    print('run_awq_true_quantflow')
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        inps.to(next(layer.parameters()).device)
        # get output as next layer's input
        # inps = layer(inps, **layer_kwargs)[0]
        layer(inps, **layer_kwargs)[0]
        
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cuda()
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        inps = layer(inps, **layer_kwargs)[0]

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results