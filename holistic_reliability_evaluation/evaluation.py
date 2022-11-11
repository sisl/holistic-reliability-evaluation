import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append('/Users/anthonycorso/Workspace/wilds/examples/')
from models.initializer import initialize_torchvision_model


import pandas as pd
import os
import numpy as np

dir = "data/"
dataset = wilds.get_dataset(dataset="camelyon17", root_dir=dir)

i = torch.randint(len(dataset), (1,))
image, y, metadata = dataset[i]
y
metadata
plt.imshow(image)

# Get the test set
transform = transforms.Compose(
    [transforms.Resize(size=(96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

test_data = dataset.get_subset("test", transform=transform)

test_loader = DataLoader(test_data, shuffle=False, sampler=None, batch_size=32)




# Load the model
state = torch.load('trained_models/camelyon17_erm_densenet121_seed0/best_model.pth', map_location=torch.device('cpu'))
state = state['algorithm']

state['model.features.denseblock4.denselayer11.norm1.running_var']

# TODO Load the algorithm?
# Pickle the configuration and then load it in here to use it to develope the algorithm
# or populate_defaults



newstate = {}
for k in state:
    newstate[k.removeprefix('model.')] = state[k]

# get constructor and last layer names:
model = initialize_torchvision_model('densenet121', 2)
model.load_state_dict(newstate)
model.eval()

input, y, metadata = next(iter(test_loader))
y_pred = model(input)

y_pred
y_pred = wilds.common.metrics.all_metrics.multiclass_logits_to_pred(y_pred)
y_pred
sum(y == y_pred) / len(y)

model(torch.zeros(1,3,96,96))

model.state_dict()['features.norm5.running_var']

model.state_dict()
