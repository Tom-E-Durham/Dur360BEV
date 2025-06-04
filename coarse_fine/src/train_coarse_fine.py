"""
Module: Train Coarse and Fine Model
Author: Wenke E (wenke.e@durham.ac.uk)
Description: This code is developed based on the PointBEV project (https://github.com/valeoai/PointBeV).
Version: 1.0
"""

from tensorboardX import SummaryWriter
import sys
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append('..')
from src import utils
from Dur360BEV_dataset import dur360bev_dataset
import time
import numpy as np
import os
from fire import Fire
import hydra
import torch
from pytorch_lightning import LightningModule


# Set up checkpoint saver:
def save_checkpoint(global_step, model, optimizer, loss,
                    checkpoint_path, scheduler=None):
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)


class BCE_Loss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(BCE_Loss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid=None):
        loss = self.loss_fn(ypred, ytgt)
        if valid is None:
            loss = reduce_masked_mean(loss, torch.ones_like(loss))
        else:
            loss = reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')

        gamma = self.gamma

        # Get the probabilities of the positive class
        pt = torch.exp(-BCE_loss)

        # Compute the Focal Loss
        F_loss = self.alpha * (1 - pt)**gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = torch.nn.functional.mse_loss(pred, gt, reduction='none')
    pos_loss = reduce_masked_mean(mse_loss, pos_mask * valid)
    neg_loss = reduce_masked_mean(mse_loss, neg_mask * valid)
    loss = (pos_loss + neg_loss) * 0.5
    return loss


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a, b) in zip(x.size(), mask.size()):
        # if not b==1:
        assert (a == b)  # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x * mask
    EPS = 1e-6
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean


def run_model(model, loss_fn, sample, device='cuda:0', sw=None):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    images = sample["image"].to(device)
    bev_seg_g = sample['bev_seg'].to(device).unsqueeze(1)
    bev_center_g = sample['center'].to(device).unsqueeze(1)
    bev_offset_g = sample['offset'].to(device).unsqueeze(1)

    images = images - 0.5  # go to -0.5, 0.5

    out = model(images)
    seg_e = out["bev"]["binimg"]
    center_e = out["bev"]["centerness"]
    offset_e = out["bev"]["offsets"]

    ce_loss = loss_fn(seg_e, bev_seg_g[:, :, 0].unsqueeze(2))  # Compute loss
    center_loss = balanced_mse_loss(center_e, bev_center_g)
    offset_loss = torch.abs(offset_e - bev_offset_g).sum(dim=2, keepdim=True)
    offset_loss = reduce_masked_mean(
        offset_loss, bev_seg_g.sum(dim=2, keepdim=True))

    ce_factor = 1 / torch.exp(model.ce_weight)
    ce_loss = ce_loss * ce_factor * 10
    ce_uncertainty_loss = 0.5 * model.ce_weight

    center_factor = 1 / (2 * torch.exp(model.center_weight))
    center_loss = center_factor * center_loss
    center_uncertainty_loss = 0.5 * model.center_weight

    offset_factor = 1 / (2 * torch.exp(model.offset_weight))
    offset_loss = offset_factor * offset_loss
    offset_uncertainty_loss = 0.5 * model.offset_weight

    total_loss += ce_loss
    total_loss += center_loss
    total_loss += offset_loss
    total_loss += ce_uncertainty_loss
    total_loss += center_uncertainty_loss
    total_loss += offset_uncertainty_loss

    # Calculating IOU
    seg_e_round = torch.sigmoid(seg_e).round()
    seg_e_car = seg_e_round[:, :, 0]  # car
    bev_seg_g_car = bev_seg_g[:, :, 0]

    # Calculate intersection and union for the car map
    intersection_car = (seg_e_car * bev_seg_g_car).sum(dim=[2, 3])
    union_car = (seg_e_car + bev_seg_g_car).clamp(0, 1).sum(dim=[2, 3])

    # Calculate IoU for the car map
    iou_car = (intersection_car / (1e-4 + union_car)).mean()

    metrics['ce_loss'] = ce_loss.item()
    metrics['center_loss'] = center_loss.item()
    metrics['offset_loss'] = offset_loss.item()
    metrics['ce_weight'] = model.ce_weight.item()
    metrics['center_weight'] = model.center_weight.item()
    metrics['offset_weight'] = model.offset_weight.item()
    metrics['iou_car'] = iou_car.item()

    if sw is not None and sw.img_save:
        print(f"[DEBUG INFO]: Saved images at step {sw.global_step}")
        sw.rgb_img('0_inputs/image', images + 0.5)
        sw.bin_img('1_outputs/bev_seg_car_e', torch.sigmoid(seg_e[:, 0, 0]))
        sw.bin_img('1_outputs/bev_seg_car_g', bev_seg_g[:, 0, 0])
        sw.bin_img('1_outputs/bev_center_e', center_e[:, 0, 0])
        sw.bin_img('1_outputs/bev_center_g', bev_center_g[:, 0, 0])
        sw.rgb_img('1_outputs/bev_offset_e', sw.offset2color(offset_e[:, 0]))
        sw.rgb_img('1_outputs/bev_offset_g',
                   sw.offset2color(bev_offset_g[:, 0]))

    return total_loss, metrics


def main(
    max_iters=4000,
    batch_size=12,
    nworkers=4,
    lr=5e-5,
    weight_decay=1e-7,
    # load_ckpt_dir=None,
    use_scheduler=True,
    dataset_dir=None,
    map_r=100,
    map_scale=2,
    log_freq=10,
    img_freq=10,
    do_val=True,
    # focal loss
    gamma=2,
    alpha=1,
):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[DEBUG INFO]: Current using device:{device}, Batch_size:{batch_size}, num_workers:{nworkers}.")

    # Create a name
    if not use_scheduler:
        model_name = f"FL{gamma}_{max_iters}_{batch_size}x{nworkers}_{lr:.0e}_{time.strftime('%m-%d_%H:%M')}"
    else:
        model_name = f"FL{gamma}_{max_iters}_{batch_size}x{nworkers}_{lr:.0e}s_{time.strftime('%m-%d_%H:%M')}"

    config_path = "../configs"
    with hydra.initialize(version_base="1.3", config_path=str(config_path)):
        cfg = hydra.compose(
            config_name="train.yaml",
            return_hydra_config=True,
        )

        cfg.paths.root_dir = str(
            pyrootutils.find_root(indicator=".project-root"))

    # Load dataset
    train_loader, val_loader = dur360bev_dataset.compile_data(dataset_dir,
                                                              batch_size=batch_size,
                                                              num_workers=nworkers,
                                                              map_r=map_r,
                                                              map_scale=map_scale,
                                                              do_shuffle=True,
                                                              is_train=True)
    train_iterloader = iter(train_loader)
    val_iterloader = iter(val_loader)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    model.to(device)
    parameters = list(model.parameters())

    # set optimizer and scheduler
    if use_scheduler:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, max_iters + 100,
                                                        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    # set the loss function
    fl_loss_fn = FocalLoss(alpha=alpha, gamma=gamma).to(device)

    for p in parameters:
        p.requires_grad = True

    # Set up the writer
    writer_t = SummaryWriter(f'../logs/coarse_fine/{model_name}/t')
    if do_val:
        writer_v = SummaryWriter(f'../logs/coarse_fine/{model_name}/v')

    # Set up checkpoints folder and checkpoints name pattern
    checkpoint_dir = f'../checkpoints/coarse_fine/{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Folder '{checkpoint_dir}' created.")
    checkpoint_pattern = 'checkpoint_epoch_{epoch}.pth'

    model.train()
    global_step = 0

    grad_acc = 5
    while global_step < max_iters:
        iter_start_time = time.time()

        global_step += 1
        iter_read_time = 0.0
        for internal_step in range(grad_acc):
            inter_start_time = time.time()

            if internal_step == grad_acc - 1:
                sw_t = utils.TensorBoardLogger(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    img_freq=img_freq
                )
            else:
                sw_t = None

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_loader)
                sample = next(train_iterloader)

            read_time = time.time() - inter_start_time
            iter_read_time += read_time

            total_loss, metrics = run_model(
                model, fl_loss_fn, sample, device, sw_t)

            total_loss.backward()

        # clean mem
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()  # Update model parameters
        if use_scheduler:
            scheduler.step()

        # Summary writer logs
        sw_t.scalar('stats/total_loss', total_loss.item())
        sw_t.scalar('stats/bev_ce_loss', metrics['ce_loss'])
        sw_t.scalar('stats/bev_center_loss', metrics['center_loss'])
        sw_t.scalar('stats/bev_offset_loss', metrics['offset_loss'])
        sw_t.scalar('stats/ce_weight', metrics['ce_weight'])
        sw_t.scalar('stats/center_weight', metrics['center_weight'])
        sw_t.scalar('stats/offset_weight', metrics['offset_weight'])
        sw_t.scalar('stats/iou_car', metrics['iou_car'])

        # run val each grad_acc step
        if do_val:
            torch.cuda.empty_cache()
            model.eval()
            sw_v = utils.TensorBoardLogger(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                img_freq=img_freq
            )
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_loader)
                sample = next(val_iterloader)

            with torch.no_grad():
                total_loss, metrics = run_model(
                    model, fl_loss_fn, sample, device, sw_v)

            # Summary writer logs
            sw_v.scalar('stats/total_loss', total_loss.item())
            sw_v.scalar('stats/bev_ce_loss', metrics['ce_loss'])
            sw_v.scalar('stats/bev_center_loss', metrics['center_loss'])
            sw_v.scalar('stats/bev_offset_loss', metrics['offset_loss'])
            sw_v.scalar('stats/iou_car', metrics['iou_car'])

            model.train()

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.scalar('_/current_lr', current_lr)

        # Save the checkpoint each 10 epoch
        if np.mod(global_step, 1000) == 0 or global_step == max_iters:
            checkpoint_path = f'{checkpoint_dir}/{checkpoint_pattern.format(epoch=global_step)}'
            save_checkpoint(global_step, model, optimizer,
                            total_loss.item(), checkpoint_path)
            print(
                f'[CHECKPOINT SAVED]: Iter: {global_step}, at {checkpoint_path}.')

        # Calculate iteration time
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        # Estimate remaining time
        remaining_time_sec = iter_time * (max_iters - global_step)
        hours, rem = divmod(remaining_time_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"[STATUS INFO]: Iter: {global_step}/{max_iters}, loss: {total_loss.item():.4f}, IoU: [Car:{metrics['iou_car']:.4f}], Iter Time: {iter_time:.2f}s, \n"
              f"Remaining Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    writer_t.close()
    if do_val:
        writer_v.close()


if __name__ == '__main__':
    Fire(main)
