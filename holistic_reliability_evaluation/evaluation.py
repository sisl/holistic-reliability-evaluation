import torch
import wilds
import numpy as np
from torchmetrics.functional import calibration_error
from torch.utils.data import DataLoader
from autoattack import AutoAttack
import pytorch_ood as ood
import torch.nn.functional as F
from wilds.common.metrics.all_metrics import multiclass_logits_to_pred
import random


# Returns the errors from the true labels and the predicted logits
def get_errors(labels, logits):
    pred = multiclass_logits_to_pred(logits)
    return labels != pred

# Defines a type of model (with potentially many random seeds)
class ModelGroup:
    def __init__(self, type, Nseeds, filepath_gen, loader, loader_kwargs, device=torch.device('cpu'), nograd_batchsize=1024, grad_batchsize=32,):
        self.type = type
        self.Nseeds = Nseeds
        self.filepath_gen = filepath_gen
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.device = device
        self.nograd_batchsize = nograd_batchsize
        self.grad_batchsize = grad_batchsize
    
    def gen_model(self, seed):
        filepath = self.filepath_gen(seed)
        return Model(self.type, seed, filepath, self.loader, self.loader_kwargs, self.device, self.nograd_batchsize, self.grad_batchsize)
    
    def gen_models(self):
        return [self.gen_model(i) for i in range(self.Nseeds)]

# Parameters for defining a conformal prediction
class ConformalPredictionParams:
    def __init__(self, calibration_set_size, alpha, conformal_score):
        self.calibration_set_size = calibration_set_size
        self.conformal_score = conformal_score
        self.alpha = alpha

# A model that can be evaluated
class Model:
    def __init__(self, type, seed, filepath, loader, loader_kwargs, device=torch.device('cpu'), nograd_batchsize=1024, grad_batchsize=32, ):
        self.type = type
        self.seed = seed
        self.filepath = filepath
        self.loader = loader
        self.loader_kwargs = loader_kwargs
        self.device = device
        self.nograd_batchsize = nograd_batchsize
        self.grad_batchsize = grad_batchsize
        self.results = dict()
    
    def name(self):
        return self.type + "-" + str(self.seed)

    def initialize_model(self):
        self.model = self.loader(self.filepath, **self.loader_kwargs)
        self.model.to(self.device).eval()
    
    def finish_model(self):
        self.model = None
        
    def init_results(self, name):
        if name not in self.results:
            self.results[name] = dict()
    
    def predict(self, dataset, name):
        loader = DataLoader(dataset, batch_size=self.nograd_batchsize, pin_memory=True)
        labels = []
        logits = []
        i = 0
        total = 0
        for x, y, metadata in loader:
            print("iteration: ", i, " total examples: ", total)
            with torch.no_grad():
                pred = self.model(x.to(self.device))
                logits.append(pred.cpu())
            labels.append(y)
            i += 1
            total += len(y)
        
        labels = torch.cat(labels)
        logits = torch.cat(logits)
        
        self.results[name]["labels"] = labels
        self.results[name]["logits"] = logits 
        return labels, logits
    
    def eval_accuracy(self, labels, logits, name):
        acc = 1 - sum(get_errors(labels, logits)).item() / logits.size(0)
        
        self.results[name]["Accuracy"] = acc
        return acc
    
    def eval_adv_accuracy(self, dataset, nexamples, name):
        loader = DataLoader(dataset, shuffle=True, batch_size=nexamples, pin_memory=True)
        adversary = AutoAttack(self.model, norm='Linf', eps=8/255, version='custom', device=self.device, attacks_to_run=['apgd-ce'])

        input, y, md = next(iter(loader))
        xadv, yadv = adversary.run_standard_evaluation(input.to(self.device), y.to(self.device), bs=self.grad_batchsize, return_labels=True)
        adv_acc = sum(y == yadv.cpu()).item() / y.size(0)
        
        self.results[name]["Adversarial Accuracy"] = adv_acc
        return adv_acc
    
    def eval_ece(self, y_true, logits, name):
        ece = calibration_error(logits.softmax(1), y_true).item()
        self.results[name]["Expected Calibration Error"] = ece
        return ece
    
    def get_ood_detector(self, ood_detector):
        if ood_detector == "Max Softmax":
            return ood.detector.MaxSoftmax(self.model)
        elif ood_detector == "Energy-Based":
            return ood.detector.EnergyBased(self.model)
        else:
            raise Exception("Didn't recognize OOD detector: ", ood_detector)
    
    def eval_ood(self, ood_detector, id_dataset, ood_dataset, name):
        detector = self.get_ood_detector(ood_detector)
        metrics = ood.utils.OODMetrics()
        id_loader = DataLoader(id_dataset, batch_size=self.nograd_batchsize, pin_memory=True)
        ood_loader = DataLoader(ood_dataset, batch_size=self.nograd_batchsize, pin_memory=True)
    
        i = 0
        total = 0
        with torch.no_grad():
            for (x, y, md) in id_loader:
                print("iteration: ", i, " total examples: ", total)
                metrics.update(detector(x.to(self.device)), y)
                i += 1
                total += len(y)
         
            for (x, y, md) in ood_loader:
                print("iteration: ", i, " total examples: ", total)
                metrics.update(detector(x.to(self.device)), -1 * torch.ones(x.shape[0]))
                i += 1
                total += len(y)
    
        ood_metrics = metrics.compute()
        self.results[name][ood_detector] = ood_metrics
        return ood_metrics
    
    def conformal_prediction(self, conf_params, labels, logits, name):
        if conf_params.conformal_score == "Softmax":
            avg_set_size = self.conf_pred_pure_smx(labels, logits, conf_params.calibration_set_size, conf_params.alpha)
        elif conf_params.conformal_score == "Acc. Softmax":
            avg_set_size = self.conf_pred_acc_smx(labels, logits, conf_params.calibration_set_size, conf_params.alpha)
        else:
            raise Exception("Unrecognized confromal score: ", conf_params.conformal_score)
        
        self.results[name][f"Avg. Set Size ({conf_params.conformal_score}, alpha={conf_params.alpha})"] = avg_set_size
        
    def conf_pred_pure_smx(self, labels, logits, cal_size, alpha):
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
        qhat = np.quantile(cal_scores, q_level, method='lower')
        prediction_sets = (val_smx >= (1-qhat))*1

        #3 get average set size
        avg_ss=0
        for set in prediction_sets:
            a = np.sum(set)
            avg_ss=avg_ss+a
        avg_ss=avg_ss/(len(prediction_sets))

        return avg_ss

    def conf_pred_acc_smx(self, labels, logits, cal_size, alpha):
        #compute and divide softmax scores
        smx = F.softmax(logits, dim=1).numpy()
        idx = np.array([1] * cal_size + [0] * (smx.shape[0]-cal_size)) > 0
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]

        #get conformal scores
        cal_pi = cal_smx.argsort(1)[:,::-1]; cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1).cumsum(axis=1) 
        cal_scores = np.take_along_axis(cal_srt,cal_pi.argsort(axis=1),axis=1)[range(cal_size),cal_labels]

        #get adjusted quantile and predicction sets for softmax predictive sets
        qhat = np.quantile(cal_scores, np.ceil((cal_size+1)*(1-alpha))/cal_size, method='higher')
        val_pi = val_smx.argsort(1)[:,::-1]; 
        val_srt = np.take_along_axis(val_smx,val_pi,axis=1).cumsum(axis=1)
        q_hat = np.zeros(shape=(len(val_srt),2))
        for i in range(len(val_srt)):
            bound = min(list(filter(lambda x: x >= qhat, val_srt[i])), default=qhat)
            q_hat[i,0]=bound
            q_hat[i,1]=bound
        prediction_sets = (np.take_along_axis(val_srt <= q_hat,val_pi.argsort(axis=1),axis=1))*1

        #get average ss
        avg_ss=0
        for set in prediction_sets:
            a = np.sum(set)
            avg_ss=avg_ss+a
        avg_ss=avg_ss/(len(prediction_sets))
        
        return avg_ss

# Function that computes the error correlations between two models for a given dataset       
def eval_error_correlation(m1, m2, dataset):
    errors1 = get_errors(m1.results[dataset]["labels"], m1.results[dataset]["logits"])
    errors2 = get_errors(m2.results[dataset]["labels"], m2.results[dataset]["logits"])
    
    # Compute the correlaion coeff: https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables
    a = errors1.logical_and(errors2).float().mean()
    b = errors1.logical_and(errors2.logical_not()).float().mean()
    c = errors1.logical_not().logical_and(errors2).float().mean()
    
    cov = a - (a+b)*(a+c)
    sigmax = torch.sqrt((a+b)*(1-(a+b)))
    sigmay = torch.sqrt((a+c)*(1-(a+c)))
    return (cov / (sigmax * sigmay)).item()

# Gets a random subset of a datset. Useful when testing the evaluation before a big run
def random_subset(dataset, numsamples, seed):
    random.seed(seed)
    indices = [i for i in range(len(dataset))]
    random_indices = random.sample(indices, numsamples)
    
    return torch.utils.data.Subset(dataset, random_indices)

# Defines which evaluations are to be done
class EvaluationSuite:
    def __init__(self, id_dataset, 
                       ds_datasets,
                       ood_datasets,
                       adv_acc = True,
                       uq_metrics = ["ece", ConformalPredictionParams(1000, 0.1, "Softmax"), ConformalPredictionParams(1000, 0.1, "Acc. Softmax")],
                       ood_detectors = ["Max Softmax", "Energy-Based"], 
                       num_adv_examples = 250, 
                       run_test=False,
                       verbose=True,
                       test_size=100, 
                       test_seed=0):
        assert len(id_dataset) == 1
        self.id_dataset = id_dataset
        self.ds_datasets = ds_datasets
        self.ood_datasets = ood_datasets
        self.adv_acc = adv_acc
        self.uq_metrics = uq_metrics
        self.ood_detectors = ood_detectors
        self.num_adv_examples = num_adv_examples
        self.run_test = run_test
        self.verbose=verbose
        self.test_size = test_size
        self.test_seed = test_seed
        
        # Set other relevant params when running a test
        if self.run_test:
            self.num_adv_examples = 1
            for uq_metric in self.uq_metrics:
                if isinstance(uq_metric, ConformalPredictionParams):
                    uq_metric.calibration_set_size=10

    def evaluate(self, model):
        # Initializes the model and moves it to the correct device
        model.initialize_model()
        
        for (name, dataset) in (self.id_dataset | self.ds_datasets).items():
            model.init_results(name)
            
            if self.run_test or "C Bar" in name:
                print("======= using smaller dataset!!")
                dataset = random_subset(dataset, self.test_size, self.test_seed)
            
            if self.verbose: print("for dataset ", name, " predicting model outputs...")
            truth, logits = model.predict(dataset, name)
            model.eval_accuracy(truth, logits, name)
            
            
            if self.adv_acc:
                if self.verbose: print("Computing adverarial accuracy...")
                model.eval_adv_accuracy(dataset, self.num_adv_examples, name)
            
            for uq_metric in self.uq_metrics:
                if self.verbose: print("Computing uq metric: ", uq_metric, "...")
                if uq_metric == "ece":
                    model.eval_ece(truth, logits, name)
                elif isinstance(uq_metric, ConformalPredictionParams):
                    model.conformal_prediction(uq_metric, truth, logits, name)
                else:
                    raise Exception("Didn't recognize UQ metric: ", uq_metric)
        
        id_dataset = next(iter(self.id_dataset.values()))
        if self.run_test:
            id_dataset = random_subset(id_dataset, self.test_size, self.test_seed)
        
        for (name, ood_dataset) in self.ood_datasets.items():
            model.init_results(name)
            for ood_detector in self.ood_detectors:
                if self.run_test:
                    ood_dataset = random_subset(ood_dataset, self.test_size, self.test_seed)
                    
                if self.verbose: print("Computing OOD detection metric on dataset ",ood_dataset, " with approach: ", ood_detector, "...")
                model.eval_ood(ood_detector, id_dataset, ood_dataset, name)
                
        # Finish with the model (frees up GPU memory)
        model.finish_model()
    
        
        