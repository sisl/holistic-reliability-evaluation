from wilds.common.metrics.all_metrics import multiclass_logits_to_pred
import random
import torch

def get_errors(labels, model_logits):
    pred = multiclass_logits_to_pred(model_logits)
    return labels != pred

def random_subset(dataset, numsamples, seed):
    random.seed(seed)
    indices = [i for i in range(len(dataset))]
    random_indices = random.sample(indices, numsamples)
    
    return torch.utils.data.Subset(dataset, random_indices)
    