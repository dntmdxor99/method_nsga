## Argument
- `model`: LlaMa model to load; pass location of Huggingface converted checkpoint.
- `dataset`: dataset to extract calibration data from (choices: "wikitext2", "ptb", "c4").
- `--seed`: seed for sampling the calibration data (default: 0).
- `--nsamples`: number of calibration data samples (default: 128).
- `--percdamp`: percent of the average Hessian diagonal to use for dampening (default: 0.01).
- `--nearest`: whether to run the RTN baseline.
- `--wbits`: number of bits to use for quantization (choices: 0, 2, 3, 4, 8, 16; default: 16).
- `--groupsize`: groupsize to use for quantization (default: -1).
- `--sym`: whether to perform symmetric quantization.
- `--save`: save quantized checkpoint under this name.
- `--new-eval`: whether to use the new PTB and C4 eval.
- `--act-order`: whether to apply the activation order GPTQ heuristic.
- `--true-sequential`: whether to run in true sequential model.
- `--static-groups`: whether to use static groups; recommended when using `--act-order` for more efficient inference.
- `--arch_path`: path to the architecture file.
- `--arch_idx`: index of the architecture (default: 0).
- `--eval`: evaluate the model (default: False).
- `--result_save_name`: name of the result file directory.

## Run
```bash
bash run_multi_gpu_with_arch.sh

or

CUDA_VISIBLE_DEVICES=0 python llama_bit_adjust_per_linear.py meta-llama/Llama-2-7b-hf c4 --wbits 0 --groupsize 0 --true-sequential --arch_path {arch_path} --arch_idx {arch_idx} --eval {True|False} --result_save_name {result_save_name}
```