import wilds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def load_camelyon17(dir, split="test", shuffle=False, batch_size=32, pin_memory=True):
    dataset = wilds.get_dataset(dataset="camelyon17", root_dir=dir)
    
    transform = transforms.Compose(
        [transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    datasubset = dataset.get_subset(split, transform=transform)
    return DataLoader(datasubset, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)

def load_camelyon17_cal(dir, split="test", shuffle=False, batch_size=32, pin_memory=True, cal_size=500):
    dataset = wilds.get_dataset(dataset="camelyon17", root_dir=dir)
    
    transform = transforms.Compose(
        [transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    datasubset = dataset.get_subset(split, transform=transform)
    cal_set, test_set = random_split(datasubset, [cal_size, len(datasubset)-cal_size])
    return DataLoader(cal_set, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory), DataLoader(test_set, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)
