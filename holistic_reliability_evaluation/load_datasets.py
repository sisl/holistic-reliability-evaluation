import wilds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_camelyon17(dir, split="test", shuffle=False, batch_size=32, pin_memory=True):
    dataset = wilds.get_dataset(dataset="camelyon17", root_dir=dir)
    
    transform = transforms.Compose(
        [transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    datasubset = dataset.get_subset(split, transform=transform)
    return DataLoader(datasubset, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)