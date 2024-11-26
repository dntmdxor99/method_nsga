# python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 0 --groupsize 0 --true-sequential
#!/bin/bash

model="meta-llama/Llama-2-7b-hf"
dataset="c4"
seed=0      ## use default
wbits=0
groupsize=0

# arch_path="/NAS/SJ/nsgaquant/save/result/2411211351_Llama-2-13b-hf_hqq_2.995_3.005/results_arch.json"
arch_path="/NAS/Woo/Automation/autoopt/archs/hqq_replace/7b/results_arch.json"
eval=True
eval_save_path="algorithm_gptq_comparing_hqq_replace_based_search"
start_gpu_id=0
end_gpu_id=3

python llama_multi_process.py $model $dataset $seed "wbits" $wbits "groupsize" $groupsize "arch_path" $arch_path "eval" $eval "eval_save_path" $eval_save_path "start_gpu_id" $start_gpu_id "end_gpu_id" $end_gpu_id