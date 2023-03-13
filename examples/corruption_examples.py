from training.corruptions import *
from training.load_datasets import *
import matplotlib.pyplot as plt


## Test individual corruptions
dataset = wilds.get_dataset(dataset="camelyon17", root_dir="/home/acorso/data")
x = next(iter(dataset))[0]

plt.imshow(x)
plt.savefig("clean.png")

xcorrupt = motion_blur(x, 1)

plt.imshow(xcorrupt/255)
plt.savefig("corrupt.png")

## Test datasets

from training.load_datasets import load_dataset
from torch.utils.data import DataLoader

dataset1 = DataLoader(load_dataset("/home/acorso/data", "camelyon17-c1-val"), batch_size=1024, num_workers=32)
dataset2 = DataLoader(load_dataset("/home/acorso/data", "camelyon17-id_val"), batch_size=1024, num_workers=32)

x = next(iter(dataset1))[0]
# x = next(iter(dataset2))[0]

x.shape

plt.imshow(x[5,...].permute(1,2,0))
plt.savefig("corrupt.png")


