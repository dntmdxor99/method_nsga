import torch
import accelerate
from statistics import median
import time
from functools import partial
from tqdm import tqdm

@torch.no_grad()
def test_latency(model, generation, device, batch_size=64, prompt_length=64, generation_length=128) :

    def cuda_timestamp(sync=False, device=None):
        if sync:
            torch.cuda.synchronize(device=device)
        return time.perf_counter()

    time_fn = partial(cuda_timestamp, device=device)

    def _step(input):
        t_step_start = time_fn()
        model(input)
        t_step_end = time_fn(True)
        return t_step_end - t_step_start

    def _step_gen(input, generation_length):
        t_step_start = time_fn()
        model.generate(input,min_new_tokens=generation_length, max_new_tokens=generation_length)
        t_step_end = time_fn(sync=True)
        return t_step_end - t_step_start
    
    latency = []
    if (generation) :
        # setting for token generation
        # generation_length = 128
        # prompt_length = 64
        # batch_size = 1
        # batch_size = 64
        max_length = prompt_length + generation_length
        model.config.max_length = max_length
        model.config.use_cache = True
        model.generation_config.use_cache = True
        iteration = 10

        # make dummy input
        random_input = torch.randint(0, 31999, (batch_size, prompt_length), dtype=torch.long)
        random_input = random_input.to(device).contiguous()

        # dummy inference
        model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)

        # latency for 10 iterations
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in tqdm(range(iteration)):
            # starter.record()
            # model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)
            # ender.record()
            # torch.cuda.synchronize()
            # cur_time = starter.elapsed_time(ender)
            cur_time = _step_gen(random_input, generation_length)
            latency.append(cur_time)

    else :
        # setting for prompt processing
        # batch_size = 1
        model.config.use_cache = False
        model.generation_config.use_cache = False
        iteration = 50

        # make dummy input for module.weight shape
        random_input = torch.randint(0, 31999, (batch_size, 2048), dtype=torch.long)
        random_input = random_input.to(device).contiguous()
        
        # dummy inference
        model(random_input)

        # latency for 50 iterations
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in tqdm(range(iteration)):
            # starter.record()
            # model(random_input)
            # ender.record()
            # torch.cuda.synchronize()
            # cur_time = starter.elapsed_time(ender)
            cur_time = _step(random_input)
            latency.append(cur_time)

    # curr_time = starter.elapsed_time(ender)
    median_latency = median(latency)
    # mean_latency = curr_time/iteration

    return median_latency
