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

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        # '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        '--wbits', type=int, default=16, choices=[0, 2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    ## customizing
    parser.add_argument('--arch_path', type=str, help='Path to the architecture file')
    parser.add_argument('--eval', type = bool_parser, default = False, help='Evaluate the model')
    parser.add_argument('--result_save_name', type=str, help='Name of the result file directory')
    parser.add_argument('--start_gpu_id', type=int, default=0, help='Start GPU ID')
    parser.add_argument('--end_gpu_id', type=int, default=0, help='End GPU ID')

    args = parser.parse_args()

    return args


def run(args, gpu_id, arch_idx):
    cmd = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'nohup',
        'python',
        '-u',
        'llama_bit_adjust_per_linear.py',

        f'{args.model}' if args.model is not None else '',
        f'{args.dataset}' if args.dataset is not None else '',
        f'--seed {args.seed}' if args.seed is not None else '',
        # f'--nsamples {args.nsamples}' if args.nsamples is not None else '',
        f'--wbits {args.wbits}' if args.wbits is not None else '',
        f'--groupsize {args.groupsize}' if args.groupsize is not None else '',
        # f'--sym' if args.sym is True else '',
        # f'--save {args.save}' if args.save is not None else '',
        # f'--new_eval' if args.new_eval is True else '',
        # f'--act_order' if args.act_order is True else '',
        f'--true-sequential' if args.true_sequential is True else '',
        # f'--static_groups' if args.static_groups is True else '',

        f'--arch_path {args.arch_path}' if args.arch_path is not None else '',
        f'--arch_idx {arch_idx}' if arch_idx is not None else '',
        f'--eval {args.eval}' if args.eval is not None else '',
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

    result_ppl_path = None
    result_sample_ppl_path = None

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
            arch_idx += 2
            if arch_idx >= len_archs:
                break
            sleep(5)

        for p in proc:
            p.join()