import os
import multiprocessing
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("start_num", type=int, default=0, help="Start number of GPU")
parser.add_argument("end_num", type=int, default=3, help="end number of GPU")
parser.add_argument('--quant_type', type=str, default='gptq', help='quantization type')

args = parser.parse_args()
if args.quant_type == 'gptq':
    large_model_path = '/SSD/Woo/gptq/llama-2-7B-INT4-g128_nsga_GPTQ'
    small_model_path = '/SSD/Woo/gptq/llama-2-7B-INT2-g64_nsga_GPTQ'
elif args.quant_type == 'hqq':
    large_model_path = '/SSD/hqq/Llama-2-7b-hf_4bit_128gs_1axis'
    small_model_path = '/SSD/hqq/Llama-2-7b-hf_2bit_64gs_1axis'
elif args.quant_type == 'awq':
    large_model_path = '/SSD/Woo/awq/llama2-7b-w4-g128-fake.pt'
    small_model_path = '/SSD/Woo/awq/llama2-7b-w2-g64-fake.pt'

# 각 프로세스에서 실행할 작업 (GPU를 사용하는 함수)
def run_on_gpu(gpu_id, flag):
    ## CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path /SSD/Woo/llama-2-7B-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-w4-g128.pt --q_backend fake
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "nohup",
        "python",
        "-u",
        "-m",
        "awq.entry",
        "--model_path",
        "/SSD/Woo/llama-2-7B-hf",
        "--w_bit",
        "4",
        "--q_group_size",
        "128",
        "--load_awq",
        "awq_cache/llama2-7b-w4-g128.pt",
        "--q_backend",
        "fake",
        "--pass_linear_list",
        "2.self_attn.q_proj 2.self_attn.k_proj 0.self_attn.v_proj 2.self_attn.v_proj 3.self_attn.v_proj",
        "--flag",
        f"{gpu_id}",
        ">",
        f"nohup_{gpu_id}.out",
        "2>&1",
        "&"
    ]

    # 명령어 실행
    print(" ".join(cmd))
    os.system(" ".join(cmd))  

if __name__ == "__main__":
    # 멀티프로세싱을 위한 프로세스 리스트
    processes = []

    # GPU 개수만큼 프로세스 생성
    for gpu_id in range(args.start_num, args.end_num + 1):
        # 각 프로세스가 실행할 GPU ID를 할당
        p = multiprocessing.Process(target=run_on_gpu, args=(gpu_id, gpu_id % 4))
        processes.append(p)
        p.start()  # 프로세스 시작
        time.sleep(3)  # 3초 대기

    # 모든 프로세스가 끝날 때까지 대기
    for p in processes:
        p.join()

    print("All processes completed.")