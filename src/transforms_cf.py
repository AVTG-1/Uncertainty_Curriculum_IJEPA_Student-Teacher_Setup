from logging import getLogger
import torch
import torchvision.transforms as transforms
from src.transforms import GaussianBlur

logger = getLogger()


def make_cifar_transforms(
    crop_size=32,
    crop_scale=(0.3, 1.0),
    color_jitter=0.5,
    horizontal_flip=True,
    color_distortion=True,
    gaussian_blur=False,
    normalization=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
):
    """
    Data augmentations tuned for CIFAR-10.
    """
    transform_list = [
        transforms.RandomResizedCrop(crop_size, scale=crop_scale),
    ]
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if color_distortion:
        cj = transforms.ColorJitter(0.8*color_jitter, 0.8*color_jitter,
                                    0.8*color_jitter, 0.2*color_jitter)
        transform_list.append(transforms.RandomApply([cj], p=0.8))
        transform_list.append(transforms.RandomGrayscale(p=0.2))
    if gaussian_blur:
        transform_list.append(GaussianBlur(p=0.5))
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1])
    ]
    transform = transforms.Compose(transform_list)
    logger.info('making CIFAR-10 data transforms')
    return transform
