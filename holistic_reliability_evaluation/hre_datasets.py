import random
import wilds
import torch
import torchvision.transforms as tfs


from torch.utils.data import random_split, Dataset, Subset

import sys, os
sys.path.append(os.path.dirname(__file__))
from corruptions import validation_corruptions, test_corruptions

# Generate a dataset that is just gaussian random noise (pre-normalized)
def random_noise_dataset(items, size, channels):
    Xtrain = torch.randn(items, channels, *size)
    ytrain = torch.randn(items, 1)
    md = torch.randn(items, 1)
    return torch.utils.data.TensorDataset(Xtrain, ytrain, md)

# Load a wilds dataset and apply the appropriate transforms
def load_wilds_dataset(
    dataset_name,
    dir,
    split,
    transforms=[],
    corruptions=[],
):
    dataset = wilds.get_dataset(dataset=dataset_name, root_dir=dir)
    transform = tfs.Compose([*corruptions, *transforms])
    return dataset.get_subset(split, transform=transform)

# Parse the dataset name and load the appropriate dataset
def load_dataset(data_dir, name, size, n_channels, transforms):
    split_words = name.split("-")
    assert len(split_words) in [1, 2, 3]
    
    dataset_name = split_words[0]
    if dataset_name == "gaussian_noise":
        return random_noise_dataset(items=10000, size=size, channels=n_channels)
    else:
        split = split_words[1]
        assert split in ["train", "val", "test", "id_val", "id_test"]
        
        corruptions = []
        if len(split_words) == 3:
            assert split_words[2] in ["corruption1_val", "corruption1_test"]
            if split_words[2] == "corruption1_val":
                corruptions=[tfs.RandomChoice(validation_corruptions(1))]
            elif split_words[2] == "corruption1_test":
                corruptions=[tfs.RandomChoice(test_corruptions(1))]
            
    # For some reason camelyon17 only has id_val and rxrx1 only has id_test
    if dataset_name in ["camelyon17", "rxrx1"] and split in ["id_val", "id_test"]:
        index = {"id_val" : 0, "id_test" : 1}[split]
        split = {"camelyon17" : "id_val", "rxrx1" : "id_test"}[dataset_name]
        dataset = load_wilds_dataset(dataset_name, data_dir, split, transforms, corruptions=corruptions)
        return random_split(dataset, [0.5, 0.5], generator=torch.Generator().manual_seed(0))[index]
    else:
        return load_wilds_dataset(dataset_name, data_dir, split, transforms, corruptions=corruptions)


# Get a subset of a provided dataset, possibly by first randomizing the indices
def get_subset(dataset, length, randomize=True):
    all_indices = [i for i in range(len(dataset))]
    if randomize:
        indices = random.sample(all_indices, length)
    else:
        indices = all_indices[:length]

    return Subset(dataset, indices)


# A dataset that combines the necessary datasets for HRE into one
class HREDatasets(Dataset):
    def __init__(
        self,
        in_distribution,
        domain_shifted,
        ood,
        length=None,
        randomize=True,
    ):
        # If length is None, set to the minmum length of any of the datasets
        if length is None:
            self.length = min(
                len(in_distribution),
                *[len(d) for d in domain_shifted],
                *[len(d) for d in ood]
            )
        # Otherwise, make sure that the length is not longer than any of the datasets
        else:
            self.length = length
            assert len(in_distribution) >= self.length
            for d in domain_shifted:
                assert len(d) >= self.length
            for d in ood:
                assert len(d) >= self.length

        # Store the dataset subsets
        self.in_distribution = get_subset(in_distribution, self.length, randomize)
        self.domain_shifted = [
            get_subset(d, self.length, randomize) for d in domain_shifted
        ]
        self.ood = [get_subset(d, self.length, randomize) for d in ood]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "id": self.in_distribution[idx],
            "ds": [d[idx] for d in self.domain_shifted],
            "ood": [d[idx] for d in self.ood],
        }
