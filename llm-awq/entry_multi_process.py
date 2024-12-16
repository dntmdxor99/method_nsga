import os
import json
import argparse
import multiprocessing
import subprocess
from datetime import datetime
from time import sleep

from manage_json import *


def get_args():
    bool_parser = lambda x : (True if x == 'True' else (False if x == 'False' else argparse.ArgumentTypeError('Boolean value expected.')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path of the hf model")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--tasks", default=None, type=str)
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
    # parser.add_argument(
    #     "--dump_awq", type=str, default=None, help="save the awq search results"
    # )
    # parser.add_argument(
    #     "--load_awq", type=str, default=None, help="load the awq search results"
    # )
    # customizing awq
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quantflow', type=bool_parser, default=False, help='If true quantization results flow')
    parser.add_argument('--bit_adjust_per_linear', type=bool_parser, default=False, help='smoothing per linear layer')
    parser.add_argument('--clip_asym', type=bool_parser, default=False, help='clip asymmetry')
    parser.add_argument('--arch_path', type=str, help='Path to the architecture file')
    parser.add_argument('--owq', type=str, default='', help='If path exists, run with owq')
    parser.add_argument(
            '--nsamples', type=int, default=128,
            help='Number of calibration data samples.'
        )
    parser.add_argument('--eval', type = bool_parser, default = False, help='Evaluate the model')
    parser.add_argument('--result_save_name', type=str, help='Name of the result file directory')
    parser.add_argument('--end_to_end', type=bool_parser, default=False, help='If true, run end to end run_awq -> quantization')

    parser.add_argument('--start_gpu_id', required=True, type=int, default=0, help='Start GPU ID')
    parser.add_argument('--end_gpu_id', required=True, type=int, default=1, help='End GPU ID')

    args = parser.parse_args()

    return args


def run(args, gpu_id, arch_idx):
    cmd = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'nohup',
        'python',
        '-u',
        '-m',
        'awq.entry',

        f'--model_path {args.model_path}' if args.model_path is not None else '',
        f'--batch_size {args.batch_size}' if args.batch_size is not None else '',
        f'--tasks {args.tasks}' if args.tasks is not None else '',
        f'--w_bit {args.w_bit}' if args.w_bit is not None else '',
        f'--q_group_size {args.q_group_size}' if args.q_group_size is not None else '',
        '--no_zero_point' if args.no_zero_point else '',
        f'--q_backend {args.q_backend}' if args.q_backend is not None else '',
        f'--dump_quant {args.dump_quant}' if args.dump_quant is not None else '',
        f'--dump_fake {args.dump_fake}' if args.dump_fake is not None else '',
        f'--load_quant {args.load_quant}' if args.load_quant is not None else '',
        '--run_awq' if args.run_awq else '',
        f'--dump_awq awq_cache/arch_{arch_idx}.pt',
        f'--load_awq awq_cache/arch_{arch_idx}.pt',
        f'--seed {args.seed}' if args.seed is not None else '',
        f'--quantflow {args.quantflow}' if args.quantflow is not None else '',
        f'--bit_adjust_per_linear {args.bit_adjust_per_linear}' if args.bit_adjust_per_linear is not None else '',
        f'--clip_asym {args.clip_asym}' if args.clip_asym is not None else '',
        f'--arch_path {args.arch_path}' if args.arch_path is not None else '',
        f'--owq {args.owq}' if args.owq is not None else
        f'--nsamples {args.nsamples}' if args.nsamples is not None else '',
        f'--eval {args.eval}' if args.eval is not None else '',
        f'--arch_idx {arch_idx}' if arch_idx is not None else '',
        f'--end_to_end {args.end_to_end}' if args.end_to_end is not None else '',

        f'--result_save_name {args.result_save_name}' if args.result_save_name is not None else '',
        
        '>',
        f'nohup_{gpu_id}.out',
        '2>&1',
    ]

    print(' '.join(cmd))
    result = subprocess.run(' '.join(cmd), shell=True)
    return result


if __name__ == "__main__":
    args = get_args()

    if args.arch_path:
        with open(args.arch_path, 'r') as f:
            data = json.load(f)
            archive = data['archive']
            len_archs = len(archive)
        assert len(archive) > 0

    
    arch_idx = 0
    while arch_idx < len_archs:
        proc = []
        for gpu_id in range(args.start_gpu_id, args.end_gpu_id + 1):
            p = multiprocessing.Process(target=run, args=(args, gpu_id, arch_idx))
            p.start()
            proc.append(p)
            arch_idx += 1
            if arch_idx >= len_archs:
                break
            sleep(5)

        for p in proc:
            p.join()

        os.system('rm -rf awq_cache/arch_*.pt')