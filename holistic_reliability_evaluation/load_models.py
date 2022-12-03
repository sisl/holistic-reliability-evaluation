import torchvision
import torch
import torch.nn as nn

def load_densenet121(path, d_out, wilds_save_format=True, device=torch.device('cpu'), prefix='model.'):
    model = torchvision.models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, d_out)
    
    state = torch.load(path, map_location=device)
    
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
    model.to(device).eval()
    return model