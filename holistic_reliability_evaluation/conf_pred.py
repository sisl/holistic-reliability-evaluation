import numpy as np
import torch
import torch.nn.functional as F
import sys 
sys.path.append('/home/dk11/holistic-reliability-evaluation/')

from holistic_reliability_evaluation.load_datasets import load_camelyon17_cal, load_camelyon17
from holistic_reliability_evaluation.load_models import load_densenet121
from holistic_reliability_evaluation.eval_functions import predict_model

nbatches=None # Set this to None if you want full evaluation
shuffle_data=False
nexamples_adv = 1 # Previously set to 250 for first round of experiments.

data_dir = "/home/dk11/data/"
device = torch.device('cuda')
out_dim = 2

# Load models
erm0 = load_densenet121("/home/dk11/trained_models/camelyon17_erm_densenet121_seed0/best_model.pth", out_dim)
erm1 = load_densenet121("/home/dk11/trained_models/camelyon17_erm_densenet121_seed1/best_model.pth", out_dim)
swav0 = load_densenet121("/home/dk11/trained_models/camelyon17_swav55_ermaugment_seed0/camelyon17_seed:0_epoch:best_model.pth", out_dim, prefix='model.0.')
swav1 = load_densenet121("/home/dk11/trained_models/camelyon17_swav55_ermaugment_seed1/camelyon17_seed:1_epoch:best_model.pth", out_dim, prefix='model.0.')

#set parameters
n=1000 #set to cal_size during full evaluation
alpha = 0.4

#load data
data = load_camelyon17(data_dir, split="test", shuffle=False, batch_size=32, pin_memory=True)
labels, logits = predict_model(swav0, data, nbatches=nbatches, device=device)
logits = torch.nan_to_num(logits)

#compute and divide softmax scores
smx = F.softmax(logits, dim=1).numpy()
idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_smx, val_smx = smx[idx,:], smx[~idx,:]
cal_labels, val_labels = labels[idx], labels[~idx]

# 1: get conformal scores. n = calib_Y.shape[0]
cal_scores = 1-cal_smx[np.arange(n),cal_labels]

# 2: get adjusted quantile and predicction sets for softmax predictive sets
q_level = np.ceil((n+1)*(1-alpha))/n
qhat = np.quantile(cal_scores, q_level, interpolation='higher')
prediction_sets = (val_smx >= (1-qhat))*1

#3 get average set size
i=0
z=0
o=0
t=0
for set in prediction_sets:
   a = np.sum(set)
   i=i+a
   if a == 0:
       z+=1
   if a == 1:
       o+=1
   if a == 2:
       t+=1


i=i/(len(prediction_sets))
z=z/(len(prediction_sets))
o=o/(len(prediction_sets))
t=t/(len(prediction_sets))

print(i)
print(z)
print(o)
print(t)
print(qhat)

# Calculate empirical coverage
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
print(f"The empirical coverage is: {empirical_coverage}")

#generate adaptive prediction sets
# Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
cal_pi = cal_smx.argsort(1)[:,::-1]; cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1).cumsum(axis=1) 
cal_scores = np.take_along_axis(cal_srt,cal_pi.argsort(axis=1),axis=1)[range(n),cal_labels]

# Get the score quantile
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

# Deploy (output=list of length n, each element is tensor of classes)
val_pi = val_smx.argsort(1)[:,::-1]; val_srt = np.take_along_axis(val_smx,val_pi,axis=1).cumsum(axis=1)
prediction_sets = np.take_along_axis(val_srt <= qhat,val_pi.argsort(axis=1),axis=1)*1

#3 get average set size
i=0
z=0
o=0
t=0
for set in prediction_sets:
   a = np.sum(set)
   i=i+a
   if a == 0:
       z+=1
   if a == 1:
       o+=1
   if a == 2:
       t+=1

i=i/(len(prediction_sets))
z=z/(len(prediction_sets))
o=o/(len(prediction_sets))
t=t/(len(prediction_sets))

print(i)
print(z)
print(o)
print(t)
print(qhat)

# Calculate empirical coverage
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
print(f"The empirical coverage is: {empirical_coverage}")