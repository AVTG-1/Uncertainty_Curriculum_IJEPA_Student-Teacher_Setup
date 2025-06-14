import os
from logging import getLogger

import torch
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

logger = getLogger()


def make_cifar10(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    data_dir=None,
    drop_last=True
):
    """
    Prepare CIFAR-10 dataloader for unsupervised I-JEPA.
    """
    # Download or point to existing
    dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        collate_fn=collator
    )
    logger.info('CIFAR-10 unsupervised loader created')
    return dataset, loader, sampler
