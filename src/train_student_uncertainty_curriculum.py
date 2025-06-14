import os
import copy
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
        filepath = os.path.join(save_dir, f"student_ep{epoch}.pth")
        torch.save(state, filepath)
        logger.info(f"Saved checkpoint: {filepath}")


def load_teacher_checkpoint(device, checkpoint_path):
    """Load teacher model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info(f"Loaded teacher checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load teacher checkpoint: {e}")
        raise


def load_student_checkpoint(device, checkpoint_path, enc, pred, opt):
    """Load student model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        enc.load_state_dict(checkpoint['enc'])
        pred.load_state_dict(checkpoint['pred'])
        opt.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
        logger.info(f"Loaded student checkpoint from {checkpoint_path}, epoch {epoch}")
        return enc, pred, opt, epoch
    except Exception as e:
        logger.error(f"Failed to load student checkpoint: {e}")
        raise


def enable_dropout_for_uncertainty(model):
    """Enable dropout layers for uncertainty estimation while keeping other layers in eval mode"""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def compute_uncertainty(student_enc, student_pred, teacher_tgt, imgs, m_e, m_p, num_samples=5, img_size=32, patch_size=4):
    """
    Compute uncertainty using input noise injection since the model has no dropout (p=0.0).
    This creates stochastic forward passes by adding small amounts of Gaussian noise to inputs.
    """
    # Set models to evaluation mode
    student_enc.eval()
    student_pred.eval()
    
    predictions = []
    
    # Get teacher target embeddings (ground truth) - deterministic
    with torch.no_grad():
        h_teacher = teacher_tgt(imgs)
        h_teacher = F.layer_norm(h_teacher, (h_teacher.size(-1),))
        h_teacher = apply_masks(h_teacher, m_p)
        h_teacher = repeat_interleave_batch(h_teacher, len(imgs), repeat=len(m_e))
    
    # Get multiple stochastic predictions from student using input noise injection
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Add small Gaussian noise to create stochastic predictions
            noise_scale = 0.1  # Small noise to avoid disrupting learning
            noisy_imgs = imgs + torch.randn_like(imgs) * noise_scale
            
            # Forward pass with noisy inputs - creates stochastic predictions
            z_student = student_enc(noisy_imgs, m_e)
            z_student = student_pred(z_student, m_e, m_p)
            predictions.append(z_student)
    
    # Stack predictions and compute variance across Monte Carlo samples
    predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, num_patches, embed_dim]
    
    # Compute uncertainty as variance across samples
    uncertainty = torch.var(predictions, dim=0)  # [batch_size, num_patches, embed_dim]
    
    # Aggregate uncertainty across embedding dimension to get per-patch uncertainty
    patch_uncertainty = torch.mean(uncertainty, dim=-1)  # [batch_size, num_patches]
    
    # Create full uncertainty maps for visualization
    total_patches = (img_size // patch_size) ** 2
    batch_size = imgs.shape[0]
    
    # Initialize full uncertainty maps with zeros
    full_uncertainty_maps = torch.zeros(batch_size, total_patches, device=imgs.device)
    
    # Fill in uncertainty values for predicted patches only
    for b in range(batch_size):
        pred_mask = m_p[0][b]  # Get prediction mask for this batch item
        # Map uncertainty values to their corresponding patch positions
        full_uncertainty_maps[b, pred_mask] = patch_uncertainty[b]
    
    # Calculate average uncertainty across all patches and batches
    avg_uncertainty = torch.mean(patch_uncertainty)
    
    # Log uncertainty statistics for debugging
    if avg_uncertainty.item() > 0:
        logger.debug(f"Uncertainty stats - Mean: {avg_uncertainty.item():.6f}, "
                    f"Min: {patch_uncertainty.min().item():.6f}, "
                    f"Max: {patch_uncertainty.max().item():.6f}, "
                    f"Std: {patch_uncertainty.std().item():.6f}")
    else:
        logger.warning(f"Zero uncertainty detected at sample {sample_idx+1}/{num_samples}. "
                      f"Predictions variance: {torch.var(predictions, dim=0).mean().item():.8f}")
    
    # Return models to full training mode
    student_enc.train()
    student_pred.train()
    
    return avg_uncertainty.item(), full_uncertainty_maps


def create_curriculum_masks(original_m_e, original_m_p, full_uncertainty_maps, difficulty_ratio=0.5):
    """
    Create curriculum-based masks focusing on patches with higher uncertainty.
    This implements hard example mining based on model uncertainty.
    """
    batch_size = full_uncertainty_maps.shape[0]
    curriculum_m_e = []
    curriculum_m_p = []
    
    for b in range(batch_size):
        # Get uncertainty for this batch item
        uncertainty = full_uncertainty_maps[b]  # [total_patches]
        
        # Get original masks for this batch item
        orig_enc_mask = original_m_e[0][b]  # Patches to keep for encoder
        orig_pred_mask = original_m_p[0][b]  # Patches to predict
        
        # Only proceed if we have uncertainty values (non-zero)
        if torch.sum(uncertainty) > 0:
            # Sort prediction patches by uncertainty (highest first)
            pred_uncertainties = uncertainty[orig_pred_mask]
            sorted_indices = torch.argsort(pred_uncertainties, descending=True)
            
            # Select most uncertain patches for curriculum learning
            num_difficult = max(1, int(len(orig_pred_mask) * difficulty_ratio))
            difficult_indices = sorted_indices[:num_difficult]
            difficult_patches = orig_pred_mask[difficult_indices]
            
            curriculum_m_e.append([orig_enc_mask])
            curriculum_m_p.append([difficult_patches])
        else:
            # Fallback to original masks if no uncertainty available
            curriculum_m_e.append([orig_enc_mask])
            curriculum_m_p.append([orig_pred_mask])
    
    # Convert to same format as original masks
    curriculum_m_e = torch.utils.data.default_collate(curriculum_m_e)
    curriculum_m_p = torch.utils.data.default_collate(curriculum_m_p)
    
    return curriculum_m_e, curriculum_m_p


def visualize_uncertainty(full_uncertainty_maps, epoch, save_dir, patch_size=4, img_size=32):
    """
    Visualize patch uncertainty as heatmaps with improved visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first few samples from batch
    num_samples = min(4, full_uncertainty_maps.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    patches_per_dim = img_size // patch_size
    
    # Calculate global min/max for consistent color scale
    global_min = full_uncertainty_maps[:num_samples].min().item()
    global_max = full_uncertainty_maps[:num_samples].max().item()
    
    for i in range(num_samples):
        uncertainty = full_uncertainty_maps[i].cpu().numpy()
        
        # Reshape to spatial dimensions
        uncertainty_map = uncertainty.reshape(patches_per_dim, patches_per_dim)
        
        # Create heatmap with consistent scale
        im = axes[i].imshow(uncertainty_map, cmap='viridis', interpolation='nearest',
                           vmin=global_min, vmax=global_max)
        axes[i].set_title(f'Sample {i+1} - Patch Uncertainty\n'
                         f'Range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]')
        axes[i].set_xlabel('Patch X')
        axes[i].set_ylabel('Patch Y')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add grid to show patch boundaries
        axes[i].set_xticks(np.arange(-0.5, patches_per_dim, 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, patches_per_dim, 1), minor=True)
        axes[i].grid(which='minor', color='white', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'uncertainty_heatmap_epoch_{epoch}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved uncertainty visualization for epoch {epoch} "
                f"(range: {global_min:.6f} - {global_max:.6f})")


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

    # Initialize student model
    student_enc, student_pred = init_model(
        device=device,
        patch_size=cfg['mask']['patch_size'],
        crop_size=cfg['data']['crop_size'],
        pred_depth=cfg['meta']['pred_depth'],
        pred_emb_dim=cfg['meta']['pred_emb_dim'],
        model_name=cfg['meta']['model_name']
    )

    # Initialize teacher model (frozen)
    teacher_enc, teacher_pred = init_model(
        device=device,
        patch_size=cfg['mask']['patch_size'],
        crop_size=cfg['data']['crop_size'],
        pred_depth=cfg['meta']['pred_depth'],
        pred_emb_dim=cfg['meta']['pred_emb_dim'],
        model_name=cfg['meta']['model_name']
    )
    
    # Load teacher checkpoint
    teacher_checkpoint = load_teacher_checkpoint(device, cfg['teacher']['checkpoint_path'])
    teacher_enc.load_state_dict(teacher_checkpoint['enc'])
    teacher_pred.load_state_dict(teacher_checkpoint['pred'])
    
    # Create teacher target encoder (EMA version)
    teacher_tgt = copy.deepcopy(teacher_enc)
    if 'tgt' in teacher_checkpoint:
        teacher_tgt.load_state_dict(teacher_checkpoint['tgt'])
    
    # Freeze teacher models
    for param in teacher_enc.parameters():
        param.requires_grad = False
    for param in teacher_pred.parameters():
        param.requires_grad = False
    for param in teacher_tgt.parameters():
        param.requires_grad = False
    
    teacher_enc.eval()
    teacher_pred.eval()
    teacher_tgt.eval()

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

    # Optimizer & schedulers for student
    opt, scaler, sched, wd_sched = init_opt(
        encoder=student_enc,
        predictor=student_pred,
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

    # Load student checkpoint if specified
    start_epoch = 1
    if cfg['meta']['load_checkpoint']:
        checkpoint_path = cfg['meta']['read_checkpoint'] or os.path.join(os.getcwd(), 'checkpoints', 'student', 'latest.pth')
        student_enc, student_pred, opt, start_epoch = load_student_checkpoint(device, checkpoint_path, student_enc, student_pred, opt)
        start_epoch += 1

    # Logger
    logdir = cfg['logging']['folder']
    os.makedirs(logdir, exist_ok=True)
    csv = CSVLogger(
        os.path.join(logdir, 'student_train.csv'),
        ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%.6f', 'uncertainty'),
        ('%.5f', 'maskA'), ('%.5f', 'maskB'), ('%d', 'time_ms')
    )

    # Training parameters
    epochs = cfg['optimization']['epochs']
    log_freq = cfg['logging']['log_freq']
    cp_freq = cfg['logging']['checkpoint_freq']
    uncertainty_freq = cfg['curriculum'].get('uncertainty_freq', 20)  # Reduced frequency for efficiency
    difficulty_ratio = cfg['curriculum'].get('difficulty_ratio', 0.5)
    use_curriculum = cfg['curriculum'].get('enabled', True)

    # Training loop
    for ep in range(start_epoch, epochs + 1):
        m = {k: AverageMeter() for k in ['loss', 'uncertainty', 'maskA', 'maskB', 'time']}
        logger.info(f"Epoch {ep}/{epochs}")
        
        # Track curriculum usage
        curriculum_used_count = 0
        total_uncertainty_computations = 0
        
        for itr, (ud, m_e, m_p) in enumerate(loader):
            imgs = ud[0].to(device)
            m_e = [x.to(device) for x in m_e]
            m_p = [x.to(device) for x in m_p]

            # Compute uncertainty and apply curriculum learning
            if itr % uncertainty_freq == 0:
                total_uncertainty_computations += 1
                uncertainty_score, full_uncertainty_maps = compute_uncertainty(
                    student_enc, student_pred, teacher_tgt, imgs, m_e, m_p,
                    num_samples=5,  # Increased for better uncertainty estimation
                    img_size=cfg['data']['crop_size'], 
                    patch_size=cfg['mask']['patch_size']
                )
                m['uncertainty'].update(uncertainty_score)
                
                # Visualize uncertainty periodically
                if itr % (uncertainty_freq * 10) == 0 and uncertainty_score > 0:
                    visualize_uncertainty(full_uncertainty_maps, ep, 
                                        os.path.join(logdir, 'uncertainty_viz'),
                                        patch_size=cfg['mask']['patch_size'],
                                        img_size=cfg['data']['crop_size'])
                
                # Apply curriculum learning if uncertainty is meaningful
                if use_curriculum and uncertainty_score > 1e-6:  # Only apply if uncertainty > threshold
                    m_e, m_p = create_curriculum_masks(m_e, m_p, full_uncertainty_maps, difficulty_ratio)
                    curriculum_used_count += 1

            def step():
                lr = sched.step()
                wd = wd_sched.step()
                
                # Get teacher target (ground truth)
                with torch.no_grad():
                    h_teacher = teacher_tgt(imgs)
                    h_teacher = F.layer_norm(h_teacher, (h_teacher.size(-1),))
                    h_teacher = apply_masks(h_teacher, m_p)
                    h_teacher = repeat_interleave_batch(h_teacher, len(imgs), repeat=len(m_e))
                
                # Student forward pass
                z_student = student_enc(imgs, m_e)
                z_student = student_pred(z_student, m_e, m_p)
                
                # Compute loss between student prediction and teacher target
                loss = F.smooth_l1_loss(z_student, h_teacher)
                
                # Backward pass
                loss.backward()
                opt.step()
                opt.zero_grad()
                
                return float(loss), lr, wd

            # Execute training step
            (loss, lr, wd), t = gpu_timer(step)
            
            # Update meters
            m['loss'].update(loss)
            m['time'].update(t)
            m['maskA'].update(len(m_e[0][0]))
            m['maskB'].update(len(m_p[0][0]))

            if itr % log_freq == 0:
                logger.info(f"It {itr+1}: loss={m['loss'].avg:.4f}, uncertainty={m['uncertainty'].avg:.6f}, "
                           f"lr={lr:.2e}, time={m['time'].avg:.1f}ms")
                csv.log(ep, itr + 1, m['loss'].avg, m['uncertainty'].avg, 
                       m['maskA'].val, m['maskB'].val, m['time'].val)

        # Save checkpoint
        state = {
            'enc': student_enc.state_dict(),
            'pred': student_pred.state_dict(),
            'opt': opt.state_dict(),
            'epoch': ep
        }
        save_checkpoint(state, os.path.join(os.getcwd(), 'checkpoints', 'student'), ep, ep == epochs, cp_freq)

        # Log epoch summary with curriculum stats
        curriculum_usage_rate = curriculum_used_count / max(total_uncertainty_computations, 1) * 100
        logger.info(f"Epoch {ep} completed - Avg Loss: {m['loss'].avg:.4f}, "
                   f"Avg Uncertainty: {m['uncertainty'].avg:.6f}, "
                   f"Curriculum Usage: {curriculum_usage_rate:.1f}% "
                   f"({curriculum_used_count}/{total_uncertainty_computations})")


if __name__ == "__main__":
    main()
