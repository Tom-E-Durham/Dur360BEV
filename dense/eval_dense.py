"""
Module: Evaluate Dense Model
Author: Wenke E (wenke.e@durham.ac.uk)
Description: This code is developed based on the SimpleBEV project (https://github.com/aharley/simple_bev/).
Version: 1.0
"""

from tensorboardX import SummaryWriter
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
    ce_loss = 10.0 * ce_loss * ce_factor
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

    # Calculating IOU
    seg_e_round = torch.sigmoid(seg_e).round()
    seg_e_car = seg_e_round[:, 0]
    bev_seg_g_car = bev_seg_g[:, 0]

    # Calculate intersection and union for the car map
    intersection_car = (seg_e_car * bev_seg_g_car).sum(dim=[1, 2])
    union_car = (seg_e_car + bev_seg_g_car).clamp(0, 1).sum(dim=[1, 2])

    # Calculate IoU 25 for the car map
    intersection_car_25 = (seg_e_car[:, 50:150, 50:150]
                           * bev_seg_g_car[:, 50:150, 50:150]).sum(dim=[1, 2])
    union_car_25 = (seg_e_car[:, 50:150, 50:150] +
                    bev_seg_g_car[:, 50:150, 50:150]).clamp(0, 1).sum(dim=[1, 2])

    # Calculate IoU 10 for the car map
    intersection_car_10 = (seg_e_car[:, 80:120, 80:120]
                           * bev_seg_g_car[:, 80:120, 80:120]).sum(dim=[1, 2])
    union_car_10 = (seg_e_car[:, 80:120, 80:120] +
                    bev_seg_g_car[:, 80:120, 80:120]).clamp(0, 1).sum(dim=[1, 2])


    metrics['ce_loss'] = ce_loss.item()
    metrics['center_loss'] = center_loss.item()
    metrics['offset_loss'] = offset_loss.item()
    metrics['ce_weight'] = model.ce_weight.item()
    metrics['center_weight'] = model.center_weight.item()
    metrics['offset_weight'] = model.offset_weight.item()
    metrics['intersection_car'] = intersection_car.sum().item()
    metrics['union_car'] = union_car.sum().item()
    metrics['intersection_car_25'] = intersection_car_25.sum().item()
    metrics['union_car_25'] = union_car_25.sum().item()
    metrics['intersection_car_10'] = intersection_car_10.sum().item()
    metrics['union_car_10'] = union_car_10.sum().item()

    if sw is not None and sw.img_save:
        print(f"[DEBUG INFO]: Saved images at step {sw.global_step}")
        sw.rgb_img('0_inputs/image', images+0.5)
        sw.bin_img('1_outputs/bev_seg_car_e', torch.sigmoid(seg_e[:, 0]))
        sw.bin_img('1_outputs/bev_seg_car_g', bev_seg_g[:, 0])
        sw.rgb_img('1_outputs/bev_offset_e', sw.offset2color(offset_e))
        sw.rgb_img('1_outputs/bev_offset_g', sw.offset2color(bev_offset_g))
        sw.bin_img('1_outputs/bev_center_e', center_e[:, 0])
        sw.bin_img('1_outputs/bev_center_g', bev_center_g[:, 0])

    else:
        pass
    return total_loss, metrics


def main(
    # evaluation
    batch_size=6,
    nworkers=6,
    checkpoint_dir=None,
    # dataset
    dataset_dir='./dur360bev_dataset/data/',
    map_r=100,
    map_scale=2,
    log_freq=1,
    img_freq=10
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[DEBUG INFO]: Current using device:{device}, Batch_size:{batch_size}, num_workers:{nworkers}.")

    # Load dataset
    _, val_loader = dur360bev_dataset.compile_data(dataset_dir,
                                                   batch_size=batch_size,
                                                   num_workers=nworkers,
                                                   map_r=map_r,
                                                   map_scale=map_scale,
                                                   do_shuffle=False,
                                                   is_train=True)
    #train_iterloader = iter(train_loader)
    val_iterloader = iter(val_loader)
    max_iters = len(val_loader)
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
    # parameters = list(model.parameters())

    # load checkpoint
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"[STATUS INFO] Total_params: {total_params}")

    loss_function = FocalLoss().to(device)

    # Set up the writer
    model_name = os.path.basename(os.path.dirname(checkpoint_dir))
    writer_v = SummaryWriter('../results/dense/{name}'.format(name=model_name))

    # Evaluation loop
    print(f"[DEBUG INFO]: Start to evaluate the model: {model_name}.")

    model.eval()
    do_val = True
    global_step = 0
    total_intersection = 0.0
    total_union = 0.0
    total_intersection_25 = 0.0
    total_union_25 = 0.0
    total_intersection_10 = 0.0
    total_union_10 = 0.0
    while do_val:
        iter_start_time = time.time()
        global_step += 1

        sw_v = utils.sw.TensorBoardLogger(
            writer=writer_v,
            global_step=global_step,
            log_freq=log_freq,
            img_freq=img_freq
        )

        try:
            sample = next(val_iterloader)
        except StopIteration:
            do_val = False

        with torch.no_grad():
            total_loss, metrics = run_model(
                model, loss_function, sample, device, sw_v)

        # Summary writer logs
        sw_v.scalar('stats/total_loss', total_loss.item())
        sw_v.scalar('stats/bev_ce_loss', metrics['ce_loss'])
        sw_v.scalar('stats/bev_center_loss', metrics['center_loss'])
        sw_v.scalar('stats/bev_offset_loss', metrics['offset_loss'])

        total_intersection += metrics['intersection_car']
        total_union += metrics['union_car']
        mean_iou = total_intersection / (1e-4 + total_union)
        sw_v.scalar('results/mean_iou', mean_iou)

        total_intersection_25 += metrics['intersection_car_25']
        total_union_25 += metrics['union_car_25']
        mean_iou_25 = total_intersection_25 / (1e-4 + total_union_25)
        sw_v.scalar('results/mean_iou_25', mean_iou_25)

        total_intersection_10 += metrics['intersection_car_10']
        total_union_10 += metrics['union_car_10']
        mean_iou_10 = total_intersection_10 / (1e-4 + total_union_10)
        sw_v.scalar('results/mean_iou_10', mean_iou_10)

        # Calculate iteration time
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        # Estimate remaining time
        remaining_time_sec = iter_time*(max_iters-global_step)
        hours, rem = divmod(remaining_time_sec, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"[STATUS INFO]: Iter: {global_step}/{max_iters}, IoU: [Car:{mean_iou:.4f}], IoU_25: [Car:{mean_iou_25:.4f}], IoU_10: [Car:{mean_iou_10:.4f}] Iter Time: {iter_time:.2f}s, \n"
              f"Remaining Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        writer_v.close()

    final_iou = total_intersection / (1e-4 + total_union)
    print(f"[STATUS INFO]: Final IoU: [Car:{final_iou:.4f}]")


if __name__ == '__main__':
    Fire(main)
