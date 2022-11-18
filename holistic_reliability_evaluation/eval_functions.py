import wilds
import wilds.common.metrics.all_metrics
import torch
from autoattack import AutoAttack
import torchmetrics

def predict_model(model, loader, nbatches=None, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    y_true = []
    model_logits = []
    i=0
    for x, y, metadata in loader:
        pred = model(x.to(device))
        model_logits.append(pred.detach().clone().to(torch.device('cpu')))
        y_true.append(y.detach().clone().to(torch.device('cpu')))
        
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
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', device='cuda', attacks_to_run=['apgd-ce'])

    input, y, md = next(iter(loader))
    xadv, yadv = adversary.run_standard_evaluation(input.to(device), y.to(device), bs=input.size(0), return_labels=True)
    return sum(y.to(device) == yadv).item() / y.size(0)


def eval_ece(y_true, model_logits):
    return torchmetrics.functional.calibration_error(model_logits.softmax(1), y_true).item()
    
    