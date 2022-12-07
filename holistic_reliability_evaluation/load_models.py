import torchvision
import torch
import torch.nn as nn

def load_densenet121(path, out_dim, wilds_save_format=True, prefix='model.'):
    model = torchvision.models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, out_dim)
    
    state = torch.load(path)
    
    if wilds_save_format:
        state = state['algorithm']
        newstate = {}
        for k in state:
            if k == 'model.1.weight':
                newstate['classifier.weight'] = state[k]
            elif k == 'model.1.bias':
                newstate['classifier.bias'] = state[k]
            else:
                newstate[k.removeprefix(prefix)] = state[k]
            
        state = newstate
    
    model.load_state_dict(state)
    return model

def load_featurized_densenet121(path, out_dim, featurizer_prefix='featurizer.', classifier_prefix='classifier.'):
    featurizer = torchvision.models.densenet121()
    featurizer_d_out = featurizer.classifier.in_features
    featurizer.classifier = nn.Identity(featurizer_d_out)

    classifier = torch.nn.Linear(featurizer_d_out, out_dim)

    state = torch.load(path)

    state = state['algorithm']
    featurizer_state = {}
    classifier_state = {}
    for k in state:
        if featurizer_prefix in k:
            featurizer_state[k.removeprefix(featurizer_prefix)] = state[k]
        elif classifier_prefix in k:
            classifier_state[k.removeprefix(classifier_prefix)] = state[k]


    featurizer.load_state_dict(featurizer_state)
    classifier.load_state_dict(classifier_state)
    return nn.Sequential(featurizer, classifier)