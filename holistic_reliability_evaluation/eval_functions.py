import wilds
import torch
from autoattack import AutoAttack


def get_vals(model, loader, nbatches=None, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    i=0
    for x, y, metadata in loader:
        y_pred_batch = model(x.to(device))
        y_pred_batch = wilds.common.metrics.all_metrics.multiclass_logits_to_pred(y_pred_batch)
        y_pred.append(y_pred_batch.detach().clone())
        y_true.append(y.detach().clone().to(device))
        
        i=i+1
        if nbatches is not None and i >= nbatches:
            break
    
    y_true = torch.cat(y_true).detach().clone().to(torch.device('cpu'))
    y_pred = torch.cat(y_pred).detach().clone().to(torch.device('cpu'))
    return y_true, y_pred

def get_errors(y_true, y_pred):
    return y_true != y_pred

def eval_accuracy(y_true, y_pred):
    return 1 - sum(get_errors(y_true, y_pred)).item() / y_true.size(0)

def eval_error_correlation(y1_true, y1_pred, y2_true, y2_pred):
    errors1 = get_errors(y1_true, y1_pred)
    errors2 = get_errors(y2_true, y2_pred)
    
    # Compute the correlaion coeff: https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables
    a = errors1.logical_and(errors2).float().mean()
    b = errors1.logical_and(errors2.logical_not()).float().mean()
    c = errors1.logical_not().logical_and(errors2).float().mean()
    
    cov = a - (a+b)*(a+c)
    sigmax = torch.sqrt((a+b)*(1-(a+b)))
    sigmay = torch.sqrt((a+c)*(1-(a+c)))
    return (cov / (sigmax * sigmay)).item()
    
    
def eval_robust_accuracy(model, loader):
    model.to(torch.device('cpu'))
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', device='cpu', attacks_to_run=['apgd-ce'])

    input, y, md = next(iter(loader))
    xadv, yadv = adversary.run_standard_evaluation(input, y, bs=input.size(0), return_labels=True)
    return eval_accuracy(y, yadv)
    
    