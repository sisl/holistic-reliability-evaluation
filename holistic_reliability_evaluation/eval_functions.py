import wilds
import wilds.common.metrics.all_metrics
import torch
from autoattack import AutoAttack
import torchmetrics
import torch.nn.functional as F
import numpy as np
import pytorch_ood as ood

def predict_model(model, loader, nbatches=None, device=torch.device('cpu')):
    y_true = []
    model_logits = []
    i=0
    for x, y, metadata in loader:
        pred = model(x.to(device))
        model_logits.append(pred.detach().clone().cpu())
        y_true.append(y.detach().clone().cpu())
        
        i=i+1
        if nbatches is not None and i >= nbatches:
            break
    
    y_true = torch.cat(y_true)
    model_logits = torch.cat(model_logits)
    return y_true, model_logits

def get_errors(y_true, model_logits):
    y_pred = wilds.common.metrics.all_metrics.multiclass_logits_to_pred(model_logits)
    return y_true != y_pred

def eval_accuracy(y_true, model_logits):
    return 1 - sum(get_errors(y_true, model_logits)).item() / y_true.size(0)

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
    
    
def eval_adv_robust_accuracy(model, loader, device=torch.device('cpu')):
    model.to(device)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', device=device.type+":"+str(device.index), attacks_to_run=['apgd-ce'])

    input, y, md = next(iter(loader))
    xadv, yadv = adversary.run_standard_evaluation(input.to(device), y.to(device), bs=32, return_labels=True)
    return sum(y.to(device) == yadv).item() / y.size(0)


def eval_ece(y_true, model_logits):
    return torchmetrics.functional.calibration_error(model_logits.softmax(1), y_true).item()


def eval_ood(detector, ID_datasets, OD_datasets, device=torch.device('cpu'), nbatches=None):
    metrics = ood.utils.OODMetrics()
    
    i=0
    for dataset in ID_datasets:
        for (x, y, md) in dataset:
            i=i+1
            metrics.update(detector(x.to(device)), y)
            if nbatches is not None and i >= nbatches:
                break
         
    i=0
    for dataset in OD_datasets:
        for (x, y, md) in dataset:
            i=i+1
            metrics.update(detector(x.to(device)), -1 * torch.ones(x.shape[0]))
            if nbatches is not None and i >= nbatches:
                break
    
    return metrics.compute()
    
def conf_pred(labels, logits, cal_size, alpha):
    #compute and divide softmax scores
    smx = F.softmax(logits, dim=1).numpy()
    idx = np.array([1] * cal_size + [0] * (smx.shape[0]-cal_size)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx,:], smx[~idx,:]
    cal_labels, val_labels = labels[idx], labels[~idx]

    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1-cal_smx[np.arange(cal_size),cal_labels]

    # 2: get adjusted quantile and predicction sets for softmax predictive sets
    q_level = np.ceil((cal_size+1)*(1-alpha))/cal_size
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    prediction_sets = (val_smx >= (1-qhat))*1

    #3 get average set size
    avg_ss=0
    for set in prediction_sets:
        a = np.sum(set)
        avg_ss=avg_ss+a
    avg_ss=avg_ss/(len(prediction_sets))

    return avg_ss


