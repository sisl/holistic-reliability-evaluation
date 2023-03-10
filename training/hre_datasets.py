import random
from torch.utils.data import Dataset, Subset

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
