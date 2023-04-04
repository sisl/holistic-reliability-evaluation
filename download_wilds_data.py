from wilds import get_dataset
import os

root_dir = 'data'


get_dataset(dataset="iwildcam", download=True, root_dir=root_dir)
get_dataset(dataset="camelyon17", download=True, root_dir=root_dir)
get_dataset(dataset="rxrx1", download=True, root_dir=root_dir)
get_dataset(dataset="fmow", download=True, root_dir=root_dir)
get_dataset(dataset="poverty", download=True, root_dir=root_dir)
get_dataset(dataset="iwildcam", download=True, root_dir=root_dir, unlabeled=True)
get_dataset(dataset="camelyon17", download=True, root_dir=root_dir, unlabeled=True)
get_dataset(dataset="fmow", download=True, root_dir=root_dir, unlabeled=True)
get_dataset(dataset="poverty", download=True, root_dir=root_dir, unlabeled=True)
