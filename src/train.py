import os
import copy
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.masks.random import MaskCollator as RandomMC, MaskCollator_cf
from src.masks.utils import apply_masks
from src.datasets.cifar10 import make_cifar10
from src.datasets.imagenet1k import make_imagenet1k
from src.transforms_cf import make_cifar_transforms
from src.transforms import make_transforms
from src.helper import init_model, init_opt
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.tensors import repeat_interleave_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def save_checkpoint(state, save_dir, epoch, is_final, checkpoint_freq):
    os.makedirs(save_dir, exist_ok=True)
    if epoch % checkpoint_freq == 0 or is_final:
        filepath = os.path.join(save_dir, f"ijepa_ep{epoch}.pth")
        torch.save(state, filepath)
        logger.info(f"Saved checkpoint: {filepath}")


# Special load_checkpoint function to handle loading of model weights of CIFAR-10 based models
# The original load_checkpoint function is designed with different labels,and is present in src/helper.py
def load_checkpoint(device, checkpoint_path, enc, pred, tgt, opt):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        enc.load_state_dict(checkpoint['enc'])
        pred.load_state_dict(checkpoint['pred'])
        tgt.load_state_dict(checkpoint['tgt'])
        opt.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
        return enc, pred, tgt, opt, epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='YAML config path')
    p.add_argument('--device', default='cuda:0', help='Compute device')
    args = p.parse_args()

    # Load YAML hyperparameters
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Seed
    seed = cfg.get('seed', 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Model + target
    enc, pred = init_model(
        device=device,
        patch_size=cfg['mask']['patch_size'],
        crop_size=cfg['data']['crop_size'],
        pred_depth=cfg['meta']['pred_depth'],
        pred_emb_dim=cfg['meta']['pred_emb_dim'],
        model_name=cfg['meta']['model_name']
    )
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad = False

    # Transforms
    if cfg['data'].get('dataset', 'imagenet1k') == 'cifar10':
        transform = make_cifar_transforms(
            crop_size=cfg['data']['crop_size'],
            crop_scale=cfg['data']['crop_scale'],
            color_jitter=cfg['data']['color_jitter_strength'],
            horizontal_flip=cfg['data']['use_horizontal_flip'],
            color_distortion=cfg['data']['use_color_distortion'],
            gaussian_blur=cfg['data']['use_gaussian_blur']
        )
    else:
        transform = make_transforms(
            crop_size=cfg['data']['crop_size'],
            crop_scale=cfg['data']['crop_scale'],
            gaussian_blur=cfg['data']['use_gaussian_blur'],
            horizontal_flip=cfg['data']['use_horizontal_flip'],
            color_distortion=cfg['data']['use_color_distortion'],
            color_jitter=cfg['data']['color_jitter_strength']
        )

    # Mask collator
    if cfg['data'].get('dataset', 'imagenet1k') == 'cifar10':
        mc = MaskCollator_cf(patch_size=cfg['mask']['patch_size'])
    else:
        mc = RandomMC(
            ratio=tuple(cfg['mask']['ratio']) if 'ratio' in cfg['mask'] else (0.4, 0.6),
            input_size=cfg['data']['crop_size'],
            patch_size=cfg['mask']['patch_size']
        )

    # DataLoader
    d = cfg['data']
    if d.get('dataset', 'imagenet1k') == 'cifar10':
        _, loader, _ = make_cifar10(
            transform=transform,
            batch_size=d['batch_size'],
            collator=mc,
            pin_mem=d['pin_mem'],
            num_workers=d['num_workers'],
            world_size=1,
            rank=0,
            data_dir=d['root_path'],
            drop_last=True
        )
    else:
        _, loader, _ = make_imagenet1k(
            transform=transform,
            batch_size=d['batch_size'],
            collator=mc,
            pin_mem=d['pin_mem'],
            training=True,
            num_workers=d['num_workers'],
            world_size=1,
            rank=0,
            root_path=d['root_path'],
            image_folder=d.get('image_folder'),
            copy_data=cfg['meta']['copy_data'],
            drop_last=True
        )

    # Optimizer & schedulers
    opt, scaler, sched, wd_sched = init_opt(
        encoder=enc,
        predictor=pred,
        wd=cfg['optimization']['weight_decay'],
        final_wd=cfg['optimization']['final_weight_decay'],
        start_lr=cfg['optimization']['start_lr'],
        ref_lr=cfg['optimization']['lr'],
        final_lr=cfg['optimization']['final_lr'],
        iterations_per_epoch=len(loader),
        warmup=cfg['optimization']['warmup'],
        num_epochs=cfg['optimization']['epochs'],
        ipe_scale=cfg['optimization']['ipe_scale'],
        use_bfloat16=cfg['meta']['use_bfloat16']
    )

    # Load checkpoint if specified
    start_epoch = 1
    if cfg['meta']['load_checkpoint']:
        checkpoint_path = cfg['meta']['read_checkpoint'] or os.path.join(os.getcwd(), 'checkpoints', 'current', 'latest.pth')
        enc, pred, tgt, opt, start_epoch = load_checkpoint(device, checkpoint_path, enc, pred, tgt, opt)
        start_epoch += 1  # Resume from next epoch

    # Momentum scheduler for target encoder
    ema = cfg['optimization']['ema']
    ipe = len(loader)
    num_epochs = cfg['optimization']['epochs']
    ipe_scale = cfg['optimization']['ipe_scale']
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    # Logger
    logdir = cfg['logging']['folder']
    os.makedirs(logdir, exist_ok=True)
    csv = CSVLogger(
        os.path.join(logdir, 'train.csv'),
        ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'),
        ('%.5f', 'maskA'), ('%.5f', 'maskB'), ('%d', 'time_ms')
    )

    # Train
    epochs = cfg['optimization']['epochs']
    log_freq = cfg['logging']['log_freq']
    cp_freq = cfg['logging']['checkpoint_freq']

    for ep in range(start_epoch, epochs + 1):
        m = {k: AverageMeter() for k in ['loss', 'maskA', 'maskB', 'time']}
        logger.info(f"Epoch {ep}/{epochs}")
        for itr, (ud, m_e, m_p) in enumerate(loader):
            imgs = ud[0].to(device)
            m_e = [x.to(device) for x in m_e]
            m_p = [x.to(device) for x in m_p]

            def step():
                lr = sched.step()
                wd = wd_sched.step()
                # Forward target
                with torch.no_grad():
                    h = tgt(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, m_p)
                    h = repeat_interleave_batch(h, len(imgs), repeat=len(m_e))
                # Forward encoder and predictor
                z = enc(imgs, m_e)
                z = pred(z, m_e, m_p)
                # Compute loss
                loss = F.smooth_l1_loss(z, h)
                loss.backward()
                opt.step()
                opt.zero_grad()
                # Update target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_e, param_t in zip(enc.parameters(), tgt.parameters()):
                        param_t.data.mul_(m).add_((1. - m) * param_e.detach().data)
                return float(loss), lr, wd

            # Execute one iteration of the training loop with GPU timer
            (loss, lr, wd), t = gpu_timer(step)
            # Update loss and time meters
            m['loss'].update(loss)
            m['time'].update(t)
            # Update mask meters
            m['maskA'].update(len(m_e[0][0]))  # Number of patches in mask A
            m['maskB'].update(len(m_p[0][0]))  # Number of patches in mask B

            if itr % log_freq == 0:
                logger.info(f"It {itr+1}: loss={m['loss'].avg:.4f}, lr={lr:.2e}, time={m['time'].avg:.1f}ms")
                csv.log(ep, itr + 1, m['loss'].avg, m['maskA'].val, m['maskB'].val, m['time'].val)

        # Checkpoint
        state = {
            'enc': enc.state_dict(),
            'pred': pred.state_dict(),
            'tgt': tgt.state_dict(),
            'opt': opt.state_dict(),
            'epoch': ep
        }
        save_checkpoint(state, os.path.join(os.getcwd(), 'checkpoints', 'current'), ep, ep == epochs, cp_freq)

if __name__ == "__main__":
    main()
