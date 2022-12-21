from operator import xor

from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch

import nv_lib.datasets
import nv_lib.collate_fn as collate_module
import nv_lib.batch_sampler as batch_sampler_module
from nv_lib.utils.parse_config import ConfigParser

def create_dataloader(dataset, configs, params):
    # select batch size or batch sampler
    assert xor("batch_size" in params, "batch_sampler" in params), \
        "You must provide batch_size or batch_sampler for each split"
    if "batch_size" in params:
        bs = params["batch_size"]
        shuffle = params.get("shuffle", True)
        batch_sampler = None
        drop_last = params.get("drop_last", True)
    elif "batch_sampler" in params:
        batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                            data_source=dataset)
        bs, shuffle = 1, False
        drop_last = False
    else:
        raise Exception()

    # Fun fact. An hour of debugging was wasted to write this line
    assert bs <= len(dataset), \
        f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

    if "collator" in params:
        collator = configs.init_obj(params["collator"], collate_module)
    else:
        collator = None
    num_workers = params.get("num_workers", 1)

    # create dataloader
    dataloader = DataLoader(
        dataset, batch_size=bs, collate_fn=collator,
        shuffle=shuffle, num_workers=num_workers,
        batch_sampler=batch_sampler, drop_last=drop_last, pin_memory=True
    )
    return dataloader


def train_test_split(dataset, test_size, seed=42):
    test_size = int(len(dataset) * test_size)
    return random_split(dataset, [len(dataset) - test_size, test_size], generator=torch.Generator().manual_seed(seed))


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(
                configs.init_obj(ds, nv_lib.datasets)
            )

        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        if split == "train_val":
            train_dataset, val_dataset = train_test_split(dataset, test_size=params["test_size"])
            split_data = {
                "train": train_dataset,
                "val": val_dataset
            }
        else:
            split_data = {
                split: dataset
            }
        
        for subsplit, subdataset in split_data.items():
            assert subsplit not in dataloaders
            print(f"Dataset {subsplit} length:", len(subdataset))
            dataloaders[subsplit] = create_dataloader(subdataset, configs, params)

    return dataloaders
