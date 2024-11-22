from tqdm import tqdm

import torch
import torch.nn as nn

from .data_utils import *

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', seqlen=2048, testloader=None, tokenizer=None):
    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer 
        )
        print(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, 1, device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, testenc, bs=1, seqlen=2048, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    n_sample = testenc.numel() // seqlen

    # List to store negative log likelihoods
    nlls = []
    # print(f"n_sample {n_sample}")

    # Loop through each batch
    for i in tqdm(range(0,n_sample,bs), desc='Eval PPL'):

        # Calculate end index
        j = min(i+bs, n_sample)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j-i, seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (n_sample * seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()

@torch.no_grad()
def eval_loss(model, testenc, bs=1, seqlen=2048, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    n_sample = testenc.numel() // seqlen
  
    # List to store negative log likelihoods
    losses = []
    
    # Loop through each batch
    for i in range(0,n_sample,bs):

        # Calculate end index
        j = min(i+bs, n_sample)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j-i, seqlen)

        # Forward pass through the model
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :]
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1)).contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * seqlen * (j-i)

        # Append to list of negative log likelihoods
        losses.append(loss)
    
    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum() / (n_sample * seqlen)
    # loss_sum = torch.stack(losses).sum() / seqlen

    return loss_sum.item()


def eval_metric(model, metric, loader, device, seqlen, bs = 1):
    if metric == 'ppl':
        return eval_ppl(model, loader, bs=bs, seqlen=seqlen, device=device)
    elif metric == 'sample_ppl':
        # return eval_loss(model, loader, bs=bs, seqlen=seqlen, device=device)
        return eval_ppl(model, loader, bs=bs, seqlen=seqlen, device=device)
    else:
        raise NotImplementedError(f'{metric} is not supported')
