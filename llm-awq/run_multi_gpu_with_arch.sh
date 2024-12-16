# python entry_multi_process.py --model_path meta-llama/Llama-2-13b-hf --w_bit 0 --q_group_size 0 --run_awq --q_backend fake --bit_adjust_per_linear --clip_asym --eval --end_to_end --arch_path /NAS/SJ/nsgaquant/save/result/2411211351_Llama-2-13b-hf_hqq_2.995_3.005/results_arch.json --seed 0 --eval_save_path test --start_gpu_id 0 --end_gpu_id 3
#!/bin/bash

model_path="meta-llama/Llama-2-7b-hf"
# batch_size=1        ## use default
# task="wikitext2"    ## use default
w_bit=0
q_group_size=0
# no_zero_point=False     ## use default
q_backend="fake"
# run_awq=True
seed=0      ## use default
quantflow=False    ## use default  
bit_adjust_per_linear=True
clip_asym=True

# arch_path="/NAS/SJ/nsgaquant/save/result/2411211351_Llama-2-13b-hf_hqq_2.995_3.005/results_arch.json"
# arch_path="/NAS/Woo/Automation/autoopt/archs/hqq_replace/7b/results_arch.json"
# arch_path="/NAS/Woo/Automation/autoopt/archs/hqq_replace/13b/results_arch.json"
# arch_path="/NAS/Woo/Automation/autoopt/archs/post_search/13b_owq/results_arch.json"
arch_path="/NAS/Woo/Automation/autoopt/archs/post_search/7b_owq/results_arch.json"

# owq="/NAS/SJ/nsgaquant/outlier/Llama-2-13b-hf/w16_r32/outlier.pth"
owq="/NAS/SJ/nsgaquant/outlier/Llama-2-7b-hf/w16_r32/outlier.pth"
# nsamples=128        ## use default
eval=True
result_save_name="algorithm_awq_based_post_search_with_owq_qs_1"
end_to_end=True
start_gpu_id=0
end_gpu_id=3

python entry_multi_process.py --model_path $model_path --w_bit $w_bit --q_group_size $q_group_size --run_awq --q_backend $q_backend --quantflow $quantflow --bit_adjust_per_linear $bit_adjust_per_linear --clip_asym $clip_asym --eval $eval --end_to_end $end_to_end --arch_path $arch_path --owq $owq --seed $seed --result_save_name $result_save_name --start_gpu_id $start_gpu_id --end_gpu_id $end_gpu_id