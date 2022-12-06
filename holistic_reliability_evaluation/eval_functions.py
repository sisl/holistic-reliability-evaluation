import wilds
import wilds.common.metrics.all_metrics
import torch
from autoattack import AutoAttack
import torchmetrics
import torch.nn.functional as F
import numpy as np
import pytorch_ood as ood

def check_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print("total: ", t, " reserved: ", r, " allocated: ", a)

def eval_error_correlation(y1_true, model1_logits, y2_true, model2_logits):
    errors1 = get_errors(y1_true, model1_logits)
    errors2 = get_errors(y2_true, model2_logits)
    
    # Compute the correlaion coeff: https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables
    a = errors1.logical_and(errors2).float().mean()
    b = errors1.logical_and(errors2.logical_not()).float().mean()
    c = errors1.logical_not().logical_and(errors2).float().mean()
    
    cov = a - (a+b)*(a+c)
    sigmax = torch.sqrt((a+b)*(1-(a+b)))
    sigmay = torch.sqrt((a+c)*(1-(a+c)))
    return (cov / (sigmax * sigmay)).item()



    



