import sys
sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar')
sys.path.append('/home/anthonycorso/Workspace/augmentation-corruption/imagenet_c_bar/utils')
import wilds
from holistic_reliability_evaluation.load_datasets import corruption_transforms
import converters
from transform_finder import transform_dict, build_transform
import torchvision.transforms
import time

data_dir = "/home/anthonycorso/Workspace/wilds/data/"

dataset = wilds.get_dataset(dataset="camelyon17", root_dir=data_dir)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(96,96)),
    torchvision.transforms.ToTensor(),
])
normal_dataset = dataset.get_subset("id_val", transform=transform)


t0 = time.time()
x, y, md = next(iter(normal_dataset))
t1 = time.time()
print("nominal: ", t1-t0)


ctransform_names = set(transform_dict.keys()).difference({'color_balance'})
ctransforms = corruption_transforms(1)

times = []
transforms_to_keep = []
for ctransform, name in zip(ctransforms, ctransform_names):
    print("running transform: ", name)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224,224)),
        converters.PilToNumpy(),
        ctransform,
        converters.NumpyToPil(),
        torchvision.transforms.Resize(size=(96,96)),
        torchvision.transforms.ToTensor(),
    ])
    c_dataset = dataset.get_subset("id_val", transform=transform)

    t0 = time.time()
    x, y, md = next(iter(c_dataset))
    t1 = time.time()
    elapsed = t1-t0
    print("took: ", elapsed)
    times.append(elapsed)
    if elapsed < 0.1:
        transforms_to_keep.append(name)

times
print(transforms_to_keep)