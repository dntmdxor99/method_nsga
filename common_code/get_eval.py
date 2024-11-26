from data_utils import get_loader
# from eval_utils import eval_metric
from eval_utils_sample_ppl_instead_loss import eval_metric
import torch


def get_eval(model, model_name, args):
    model = model.to(torch.device("cuda"))
    model.seqlen = 2048
  
    print('loading data...')
    # datasets = ['wikitext2', 'c4']
    datasets = ['wikitext2']
    train_loaders = {dataset: get_loader(dataset, model=model_name, n_sample=args.nsamples , train=True, seed=args.seed, seqlen=model.seqlen) for dataset in datasets}
    test_loaders = {dataset: get_loader(dataset, model=model_name, train=False, seqlen=model.seqlen) for dataset in datasets}

    metric_ppl = dict()

    for metric in ['ppl', 'sample_ppl']:
        if metric == 'ppl':
            loaders = test_loaders
        elif metric == 'sample_ppl':
            loaders = train_loaders
        else:
            NotImplementedError(f'metric should be ppl or sample_ppl, but got {metric}')

        metric_list = dict()
        for dataset, loader in loaders.items():
            metric_list[dataset] = eval_metric(model=model, metric=metric, loader=loader, device=torch.device("cuda"), seqlen=model.seqlen)
            print(f'{dataset} {metric} : {metric_list[dataset]}')

        metric_ppl[metric] = metric_list

    return metric_ppl