## Argument
- `--model_path`: path of the hf model
- `--batch_size`: batch size (default: 1)
- `--tasks`: tasks (default: None)
- `--output_path`: output path (default: None)
- `--num_fewshot`: number of few-shot examples (default: 0)
- `--parallel`: enable model parallelism
- `--max_memory`: list of device_id:max_memory pairs to be parsed into a dictionary; Example: 0:10GiB 1:10GiB cpu:30GiB; more details here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
- `--auto_parallel`: automatically set parallel and batch_size
- `--w_bit`: weight bit-width (default: None)
- `--q_group_size`: quantization group size (default: -1)
- `--no_zero_point`: disable zero_point
- `--q_backend`: quantization backend (choices: "fake", "real"; default: "fake")
- `--dump_quant`: save quantized model
- `--dump_fake`: save fake-quantized model
- `--load_quant`: load quantized model
- `--run_awq`: perform awq search process
- `--dump_awq`: save the awq search results
- `--load_awq`: load the awq search results
- `--vila-15`: quantizing vila 1.5
- `--seed`: random seed (default: 0)
- `--quantflow`: True quantization result flow
- `--bit_adjust_per_linear`: smoothing per linear layer
- `--clip_asym`: clip asymmetry
- `--arch_path`: path to the architecture file
- `--nsamples`: number of calibration data samples (default: 128)
- `--eval`: evaluate the model
- `--arch_idx`: index of the architecture (default: 0)
- `--end_to_end`: if true, run end to end run_awq -> quantization
- `--result_ppl_path`: path to the result file
- `--result_sample_ppl_path`: path to the result sample file


## Run
```bash
bash run_multi_gpu_with_arch.sh

or

CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path meta-llama/Llama-2-13b-hf --batch_size 1  --w_bit 0 --q_group_size 0  --q_backend fake    --run_awq --dump_awq awq_cache/arch_0.pt --load_awq awq_cache/arch_0.pt --seed 0 --quantflow False --bit_adjust_per_linear True --clip_asym True --arch_path {arch_path.json} --nsamples 128 --eval True --arch_idx 0 --end_to_end True --result_ppl_path {result_ppl_path.json} --result_sample_ppl_path {result_sample_ppl_path.json}
```