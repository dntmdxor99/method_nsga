# Import necessary modules
from tqdm import tqdm
import os

import torch
import torch.nn as nn

# Import get_loaders function from data module within the same director
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

def get_tokenizer(model, cache_dir=None):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    return tokenizer

def get_wikitext2(tokenizer, cache_dir=None):
    
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=cache_dir)

    # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    return testenc
    # random.seed(seed)
    # trainloader = []
    # for _ in range(n_sample):
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    # new_trainloader = []
    # num_batches = n_sample // batch_size + (int)(n_sample % batch_size > 0)
    # for i in range(0, num_batches):
    #     start =  i * batch_size
    #     end = min(start + batch_size, n_sample)
    #     batched_inp = []
    #     batched_tar = []
    #     for j in range(start, end):
    #         batched_inp.append(trainloader[j][0])
    #         batched_tar.append(trainloader[j][1])
    #     batched_inp = torch.cat(batched_inp)
    #     batched_tar = torch.cat(batched_tar)
    #     new_trainloader.append((batched_inp, batched_tar))
    # del trainloader
    # trainloader = new_trainloader
    # del new_trainloader

    # return trainloader, testenc

def get_c4(seqlen,tokenizer, cache_dir=None):
   
    # traindata = load_dataset(
    #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir
    # )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)

    # random.seed(seed)
    # trainloader = []
    # for _ in range(n_sample):
    #     while True:
    #         i = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] > seqlen:
    #             break
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    # new_trainloader = []
    # num_batches = n_sample // batch_size + (int)(n_sample % batch_size > 0)
    # for i in range(0, num_batches):
    #     start =  i * batch_size
    #     end = min(start + batch_size, n_sample)
    #     batched_inp = []
    #     batched_tar = []
    #     for j in range(start, end):
    #         batched_inp.append(trainloader[j][0])
    #         batched_tar.append(trainloader[j][1])
    #     batched_inp = torch.cat(batched_inp)
    #     batched_tar = torch.cat(batched_tar)
    #     new_trainloader.append((batched_inp, batched_tar))
    # del trainloader
    # trainloader = new_trainloader
    # del new_trainloader

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)

    return valenc
    # return trainloader, valenc

def get_wikitext2_trainenc(seed, n_sample, tokenizer, cache_dir=None):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt')

    return trainenc

def get_c4_trainenc(seed, n_sample, tokenizer, cache_dir=None):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir
    )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)
    
    trainenc = tokenizer(' '.join(traindata[:n_sample]['text']), return_tensors='pt')
    trainenc = trainenc.input_ids

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    trainenc = TokenizerWrapper(trainenc)

    return trainenc

def get_trainloaders(name, n_sample=128, seed=0, seqlen=2048, model='', batch_size=1, cache_dir=None):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'c4' in name:
        return get_c4_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)

def get_loader(name, n_sample=128, train=True, seed=0, seqlen=2048, tokenizer=None, model='', cache_dir=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    if train:
        if 'wikitext2' in name:
            return get_wikitext2_trainenc(seed=seed, n_sample=n_sample, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4_trainenc(seed=seed, n_sample=n_sample, tokenizer=tokenizer, cache_dir=cache_dir)
    else:
        if 'wikitext2' in name:
            return get_wikitext2(tokenizer=tokenizer, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4(seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
