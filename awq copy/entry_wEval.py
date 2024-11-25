from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
import sys
import random
import numpy as np


from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    pseudo_quantize_model_weight_wBit_setting,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

from search_space.llama import LlamaSearchSpace

from awq.utils.data_utils import get_loader
# from utils.eval_utils import eval_metric
from awq.utils.eval_utils_sample_ppl_instead_loss import eval_metric

# import nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_fake", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--true-quantflow', action='store_true', help='True quantization result flow')
parser.add_argument('--bit_diff_while_smoothing', action='store_true', help='Bit configuration while getting smoothing factor')
parser.add_argument('--per_linear', action='store_true', help='smoothing per linear layer')
parser.add_argument('--arch_path', type=str, help='Path to the architecture file')
parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )


args = parser.parse_args()

vila_10_quant_mode = ("llava" in args.model_path.lower() or "vila" in args.model_path.lower()) and not args.vila_15

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)


def build_model_and_enc(model_path, archs = None, search_space_config = None, flag = None):
    # if not os.path.exists(model_path):  # look into ssd
    #     raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False}
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()
        
        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            # """
            # SmoothQuant 적용해보기
            # """
            # print("-----\nApply smoothquant\n-----")
            # act_scales = torch.load("/NAS/Woo/smoothquant/act_scales/llama-3-8b-hf")
            # smooth_lm(model, act_scales, 0.85)

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
                
                ## customizing
                true_quantflow = args.true_quantflow,
                bit_diff_while_smoothing = args.bit_diff_while_smoothing,
                arch = None if not args.bit_diff_while_smoothing else archs,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)
            # print(create_2_bit)
            exit(0)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config, 
                                             ## customizing
                                             bit_diff_while_smoothing = args.bit_diff_while_smoothing,
                                             arch = None if not args.bit_diff_while_smoothing else archs,
                                             )
                # pseudo_quantize_model_weight_wBit_setting(model, w_bit=args.w_bit, q_config=q_config, archs = archs, search_space_config=search_space_config, flag=flag)
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()


    if args.bit_diff_while_smoothing:
        import json
        with open(args.arch_path, 'r') as f:
            arch = json.load(f)
            archs = arch['archive']
    else:
        archs = None
    model, enc = build_model_and_enc(args.model_path, archs)

    model = model.to(torch.device("cuda"))
    model.seqlen = 2048
  
    print('loading data...')
    # datasets = ['wikitext2', 'c4']
    datasets = ['wikitext2']
    train_loaders = {dataset: get_loader(dataset, model=args.model_path, n_sample=args.nsamples , train=True, seed=args.seed, seqlen=model.seqlen) for dataset in datasets}
    test_loaders = {dataset: get_loader(dataset, model=args.model_path, train=False, seqlen=model.seqlen) for dataset in datasets}

    for metric in ['ppl', 'sample_ppl']:
        if metric == 'ppl':
            loaders = test_loaders
        elif metric == 'sample_ppl':
            loaders = train_loaders
        else:
            NotImplementedError(f'metric should be ppl or sample_ppl, but got {metric}')

        metric_list = dict()
        for dataset, loader in loaders.items():
            metric_list[dataset] = eval_metric(model=model, metric=metric, loader=loader, device=torch.device("cuda"), seqlen=model.seqlen)
            print(f'{dataset} {metric} : {metric_list[dataset]}')

if __name__ == "__main__":
    main()