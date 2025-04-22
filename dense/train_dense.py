"""
Module: Train Dense Model
Author: Wenke E (wenke.e@durham.ac.uk)
Description: This code is developed based on the SimpleBEV project (https://github.com/aharley/simple_bev/).
Version: 1.0
"""

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch
import utils.sw
import utils.vox
from nets.df_segnet import Segnet, FocalLoss
from Dur360BEV_dataset import dur360bev_dataset
import numpy as np
import time
import os
from fire import Fire
import sys
sys.path.append('..')


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
                                                    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = torch.nn.functional.mse_loss(pred, gt, reduction='none')
    pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss


# Set up checkpoint saver:
def save_checkpoint(global_step, model, optimizer, loss, checkpoint_path, scheduler=None):
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)


def run_model(model, loss_fn, sample, device='cuda:0', sw=None):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    # Ensure the data is on the correct device
    images = sample['image'].to(device)
    bev_seg_g = sample['bev_seg'].to(device)
    bev_center_g = sample['center'].to(device)
    bev_offset_g = sample['offset'].to(device)

    images = images - 0.5  # go to -0.5, 0.5

    raw_e, feat_e, seg_e, center_e, offset_e = model(
        images)  # Pass the batch of inputs through the model
    #print(f"[DEBUG INFO]: g: {bev_seg_g.shape}, e: {seg_e.shape}")
    ce_loss = loss_fn(seg_e, bev_seg_g[:, 0].unsqueeze(1))  # Compute loss
    center_loss = balanced_mse_loss(center_e, bev_center_g)
    offset_loss = torch.abs(offset_e-bev_offset_g).sum(dim=1, keepdim=True)

    offset_loss = utils.basic.reduce_masked_mean(
        offset_loss, bev_seg_g.sum(dim=1, keepdim=True))

    ce_factor = 1 / torch.exp(model.ce_weight)
    ce_loss = ce_loss * ce_factor * 10
    ce_uncertainty_loss = 0.5 * model.ce_weight

    center_factor = 1 / (2*torch.exp(model.center_weight))
    center_loss = center_factor * center_loss
    center_uncertainty_loss = 0.5 * model.center_weight

    offset_factor = 1 / (2*torch.exp(model.offset_weight))
    offset_loss = offset_factor * offset_loss
    offset_uncertainty_loss = 0.5 * model.offset_weight

    total_loss += ce_loss
    total_loss += center_loss
    total_loss += offset_loss
    total_loss += ce_uncertainty_loss
    total_loss += center_uncertainty_loss
    total_loss += offset_uncertainty_loss

    seg_e_round = torch.sigmoid(seg_e).round()

    # Separate the car and pedestrian maps and bin_map
    seg_e_car = seg_e_round[:, 0]  # car
    bev_seg_g_car = bev_seg_g[:, 0]

    # Calculate intersection and union for the car map
    intersection_car = (seg_e_car * bev_seg_g_car).sum(dim=[1, 2])
    union_car = (seg_e_car + bev_seg_g_car).clamp(0, 1).sum(dim=[1, 2])

    # Calculate IoU for the car map
    iou_car = (intersection_car / (1e-4 + union_car)).mean()

    metrics['ce_loss'] = ce_loss.item()
    metrics['center_loss'] = center_loss.item()
    metrics['offset_loss'] = offset_loss.item()
    metrics['ce_weight'] = model.ce_weight.item()
    metrics['center_weight'] = model.center_weight.item()
    metrics['offset_weight'] = model.offset_weight.item()
    metrics['iou_car'] = iou_car.item()
    if loss_fn.trainable:
        metrics['FL_gamma'] = F.softplus(loss_fn.gamma_param).item()
    else:
        metrics['FL_gamma'] = loss_fn.gamma

    if sw is not None and sw.img_save:
        print(f"[DEBUG INFO]: Saved images at step {sw.global_step}")
        sw.rgb_img('0_inputs/image', images+0.5)
        sw.bin_img('1_outputs/bev_seg_car_e',
                   torch.sigmoid(seg_e[:, 0]).round())
        sw.bin_img('1_outputs/bev_seg_car_g', bev_seg_g[:, 0])
        sw.rgb_img('1_outputs/bev_offset_e', sw.offset2color(offset_e))
        sw.rgb_img('1_outputs/bev_offset_g', sw.offset2color(bev_offset_g))
        sw.bin_img('1_outputs/bev_center_e', center_e[:, 0])
        sw.bin_img('1_outputs/bev_center_g', bev_center_g[:, 0])
    else:
        pass
    return total_loss, metrics


def main(
    # training
    max_iters=25000,
    batch_size=6,
    nworkers=6,
    lr=5e-5,
    gamma=2,
    weight_decay=1e-7,
    load_ckpt_dir=None,
    use_scheduler=True,
    # dataset
    dataset_dir='./dur360bev_dataset/data/',
    map_r=100,
    map_scale=2,
    log_freq=10,
    img_freq=100,
    do_val=False
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[DEBUG INFO]: Current using device:{device}, Batch_size:{batch_size}, num_workers:{nworkers}.")

    # Create a name
    if not use_scheduler:
        model_name = f"{max_iters}_{batch_size}x{nworkers}_{lr:.0e}_{time.strftime('%m-%d_%H:%M')}"
    else:
        model_name = f"{max_iters}_{batch_size}x{nworkers}_{lr:.0e}s_{time.strftime('%m-%d_%H:%M')}"

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
    # Prepare the model
    scene_centroid_x = 0.0
    scene_centroid_y = 1.0  # down 1 meter
    scene_centroid_z = 0.0

    scene_centroid_py = np.array([scene_centroid_x,
                                  scene_centroid_y,
                                  scene_centroid_z]).reshape([1, 3])

    scene_centroid = torch.from_numpy(scene_centroid_py).float()
    XMIN, XMAX = -50, 50
    ZMIN, ZMAX = -50, 50
    YMIN, YMAX = -5, 5
    bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

    Z, Y, X = 200, 8, 200

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    model = Segnet(Z, Y, X, vox_util, rand_flip=True, encoder_type='res101')
    model = model.to(device)
    parameters = list(model.parameters())

    # Set optimizer and scheduler
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr,
                                               weight_decay,
                                               1e-8,
                                               max_iters,
                                               model.parameters())
    else:
        optimizer = torch.optim.Adam(
            parameters, lr=lr, weight_decay=weight_decay)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"[STATUS INFO] Total_params: {total_params}")

    loss_function = FocalLoss(trainable=False, gamma=gamma).to(device)

    requires_grad(parameters, True)

    # Set up the writer
    writer_t = SummaryWriter(f'../logs/dense/FL{gamma}_{model_name}/t')
    if do_val:
        writer_v = SummaryWriter(f'../logs/dense/FL{gamma}_{model_name}/v')

    # Set up checkpoints folder and checkpoints name pattern
    checkpoint_dir = f'../checkpoints/dense/FL{gamma}_{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Folder '{checkpoint_dir}' created.")
    checkpoint_pattern = 'checkpoint_epoch_{epoch}.pth'

    # Restore the model and optimizer states
    print(f"[DEBUG INFO]: Load checkpoint from {load_ckpt_dir}.")
    if load_ckpt_dir is not None:
        checkpoint = torch.load(load_ckpt_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        loss = checkpoint['loss']
        if use_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        global_step = 0

    # Training loop
    print(
        f"[DEBUG INFO]: Start to train {max_iters-global_step} steps from the global step: {global_step}.")

    model.train()

    grad_acc = 5
    while global_step < max_iters:
        iter_start_time = time.time()

        global_step += 1
        iter_read_time = 0.0
        for internal_step in range(grad_acc):
            inter_start_time = time.time()

            if internal_step == grad_acc-1:
                sw_t = utils.sw.TensorBoardLogger(
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

            read_time = time.time()-inter_start_time
            iter_read_time += read_time

            total_loss, metrics = run_model(
                model, loss_function, sample, device, sw_t)

            total_loss.backward()

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
        sw_t.scalar('_/FL_gamma', metrics['FL_gamma'])

        # run val each 5 step
        if do_val:
            torch.cuda.empty_cache()
            model.eval()
            sw_v = utils.sw.TensorBoardLogger(
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
                    model, loss_function, sample, device, sw_v)

            # Summary writer logs
            sw_v.scalar('stats/total_loss', total_loss.item())
            sw_v.scalar('stats/bev_ce_loss', metrics['ce_loss'])
            sw_v.scalar('stats/bev_center_loss', metrics['center_loss'])
            sw_v.scalar('stats/bev_offset_loss', metrics['offset_loss'])
            sw_v.scalar('stats/iou_car', metrics['iou_car'])
            #sw_v.scalar('stats/iou_pedestrian', metrics['iou_pedestrian'])
            #sw_v.scalar('stats/iou_map', metrics['iou_map'])

            model.train()

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.scalar('_/current_lr', current_lr)

        # Save the checkpoint each 1000 iterations
        if np.mod(global_step, 1000) == 0 or global_step == max_iters:
            checkpoint_path = f'{checkpoint_dir}/{checkpoint_pattern.format(epoch=global_step)}'
            save_checkpoint(global_step, model, optimizer,
                            total_loss.item(), checkpoint_path, scheduler)
            print(
                f'[CHECKPOINT SAVED]: Iter: {global_step}, at {checkpoint_path}.')

        # Calculate iteration time
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        # Estimate remaining time
        remaining_time_sec = iter_time*(max_iters-global_step)
        hours, rem = divmod(remaining_time_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"[STATUS INFO]: Iter: {global_step}/{max_iters}, loss: {total_loss.item():.4f}, IoU: [Car:{metrics['iou_car']:.4f}], Iter Time: {iter_time:.2f}s, \n"
              f"Remaining Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    writer_t.close()
    if do_val:
        writer_v.close()


if __name__ == '__main__':
    Fire(main)
