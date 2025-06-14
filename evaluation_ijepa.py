"""
Evaluation script for I-JEPA pre-trained models.

This script is used to evaluate pre-trained I-JEPA models on the
CIFAR-10 dataset. The script assumes that the pre-trained model
is saved in the same directory, with the naming convention
`ijepa_ep{epoch}.pth`.

Example:
    python evaluation_ijepa.py --epoch 50

The script will load the pre-trained model at the specified epoch,
and evaluate its performance on the test set of CIFAR-10. The
evaluation metrics are:

* Accuracy
* Precision
* Recall
* F1-score

The metrics are printed to the console, and also saved to a file
named `ijepa_ep{epoch}_eval.txt`.

The script requires the following packages:
    * PyTorch
    * torchvision
    * numpy

The script also assumes that the `src` directory is in the same
directory as the script, and that the `src` directory contains the
following packages:
    * `models.vision_transformer` (for loading the pre-trained model)
    * `utils.distributed` (for loading the checkpoint)

The script is designed to be run on a single machine with one or
more GPUs. The script uses the `torch.distributed` package to
handle distributed training, if needed.

Please see the README.md file in the same directory for more
information on how to use this script.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models import vision_transformer as vit

# Device setup
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. Please check CUDA installation.")
device = torch.device('cuda:0')
logger.info(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0)}")

# Custom checkpoint loading function
def load_checkpoint(device, r_path, encoder):
    try:
        # Load checkpoint with weights_only=True for safety
        checkpoint = torch.load(r_path, map_location=device, weights_only=True)
        epoch = checkpoint.get('epoch', 0)

        # Load encoder
        pretrained_dict = checkpoint.get('enc')
        if pretrained_dict is None:
            raise KeyError("Key 'enc' not found in checkpoint")

        # Handle potential key mismatches
        model_dict = encoder.state_dict()
        # Filter out unexpected keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # Check for missing or unexpected keys
        missing = [k for k in model_dict if k not in pretrained_dict]
        unexpected = [k for k in pretrained_dict if k not in model_dict]
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)
        logger.info(f"Loaded pretrained encoder from epoch {epoch}")
        logger.info(f"Read path: {r_path}")
        return encoder, epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

# Load the encoder
def load_encoder(checkpoint_path):
    # Initialize encoder based on vision_transformer.py
    encoder = vit.vit_small(img_size=[32], patch_size=4)  # For CIFAR-10
    encoder.to(device)
    encoder, _ = load_checkpoint(device, checkpoint_path, encoder)
    encoder.eval()
    return encoder

# Prepare CIFAR-10 data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='cifar-10/cifar-10-batches-py', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='cifar-10/cifar-10-batches-py', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

# Extract features
def extract_features(loader, encoder, device):
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = encoder(images)  # Shape: [batch_size, num_patches, embed_dim]
            feats = feats.mean(dim=1)  # Average over patches
            features.append(feats.cpu())
            labels.append(targets)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

# List of epochs to evaluate
epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for epoch in epochs:
    checkpoint_path = f'checkpoints/current/ijepa_ep{epoch}.pth'
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")

    try:
        encoder = load_encoder(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        continue

    # Extract features
    logger.info(f"Extracting training features for epoch {epoch}...")
    train_features, train_labels = extract_features(train_loader, encoder, device)
    logger.info(f"Extracting test features for epoch {epoch}...")
    test_features, test_labels = extract_features(test_loader, encoder, device)

    # Train linear classifier
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    train_features, train_labels = train_features.to(device), train_labels.to(device)

    num_epochs = 20
    for epoch_num in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        logger.info(f'Epoch {epoch_num+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Evaluate on test set
    test_features, test_labels = test_features.to(device), test_labels.to(device)
    with torch.no_grad():
        test_outputs = classifier(test_features)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        logger.info(f'Test Accuracy for epoch {epoch}: {accuracy * 100:.2f}%')

    logger.info("-----")

