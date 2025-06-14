# I-JEPA

This repository is organized as follows:

- `checkpoints/`: contains the pre-trained models (teacher and student) used for evaluation.
- `configs/`: contains the hyperparameters used for training and evaluation.
- `logs/`: contains the logs of the training process.
- `src/`: contains the source code for the I-JEPA model.
  - `datasets/`: contains the code for loading the data (CIFAR-10 and ImageNet).
  - `helper.py`: contains functions for loading the model, optimizer, and scheduler.
  - `masks/`: contains the code for the mask generation (random and center-crop).
  - `models/`: contains the code for the I-JEPA model.
  - `transforms/`: contains the code for the data transformations.
  - `train.py`: contains the code for training the I-JEPA model.
  - `utils/`: contains the code for the logging and distributed training.
- `evaluation_ijepa.py`: contains the code for evaluating the pre-trained models.

To run the training and evaluation scripts, use the following commands:

- `PYTHONPATH=. python src/train.py --config configs/cifar10_vit_small_ep_n.yaml --device cuda:0` to train the I-JEPA model.
- `PYTHONPATH=. python evaluation_ijepa.py` to evaluate the pre-trained model.
