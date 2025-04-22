"""
Module: Evaluate Coarse and Fine Model
Author: Wenke E (wenke.e@durham.ac.uk)
Description: This code is developed based on the PointBEV project (https://github.com/valeoai/PointBeV).
Version: 1.0
"""


import sys
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append('..')
from Dur360BEV_dataset import dur360bev_dataset
from src import utils
from tensorboardX import SummaryWriter
import time
import os
from fire import Fire
import hydra
import torch
from pytorch_lightning import LightningModule




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

    # Calculate IoU 25 for the car map
    intersection_car_25 = (seg_e_car[:, :, 50:150, 50:150]
                           * bev_seg_g_car[:, :, 50:150, 50:150]).sum(dim=[2, 3])
    union_car_25 = (seg_e_car[:, :, 50:150, 50:150] +
                    bev_seg_g_car[:, :, 50:150, 50:150]).clamp(0, 1).sum(dim=[2, 3])

    # Calculate IoU 10 for the car map
    intersection_car_10 = (seg_e_car[:, :, 80:120, 80:120]
                           * bev_seg_g_car[:, :, 80:120, 80:120]).sum(dim=[2, 3])
    union_car_10 = (seg_e_car[:, :, 80:120, 80:120] +
                    bev_seg_g_car[:, :, 80:120, 80:120]).clamp(0, 1).sum(dim=[2, 3])

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
    batch_size=12,
    nworkers=4,
    checkpoint_dir=None,
    dataset_dir=None,
    map_r=100,
    map_scale=2,
    log_freq=1,
    img_freq=10,
    gamma=1
):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[DEBUG INFO]: Current using device:{device}, Batch_size:{batch_size}, num_workers:{nworkers}.")

    config_path = "../configs"
    with hydra.initialize(version_base="1.3", config_path=str(config_path)):
        cfg = hydra.compose(
            config_name="train.yaml",
            return_hydra_config=True,
        )

        cfg.paths.root_dir = str(
            pyrootutils.find_root(indicator=".project-root"))

    _, val_loader = dur360bev_dataset.compile_data(dataset_dir,
                                                   batch_size=batch_size,
                                                   num_workers=nworkers,
                                                   map_r=map_r,
                                                   map_scale=map_scale,
                                                   do_shuffle=False,
                                                   is_train=True)
    val_iterloader = iter(val_loader)
    max_iters = len(val_loader)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    model.to(device)
    # parameters = list(model.parameters())

    # load checkpoint
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    # set the loss function
    fl_loss_fn = FocalLoss(gamma=gamma).to(device)

    # Set up the writer
    model_name = os.path.basename(os.path.dirname(checkpoint_dir))
    writer_v = SummaryWriter(f'../results/coarse_fine/{model_name}')

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
                model, fl_loss_fn, sample, device, sw_v)

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
        remaining_time_sec = iter_time * (max_iters - global_step)
        hours, rem = divmod(remaining_time_sec, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"[STATUS INFO]: Iter: {global_step}/{max_iters}, IoU: [Car:{mean_iou:.4f}], IoU_25: [Car:{mean_iou_25:.4f}], IoU_10: [Car:{mean_iou_10:.4f}] Iter Time: {iter_time:.2f}s, \n"
              f"Remaining Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        writer_v.close()

    final_iou = total_intersection / (1e-4 + total_union)
    print(f"[STATUS INFO]: Final IoU: [Car:{final_iou:.4f}]")


if __name__ == '__main__':
    Fire(main)
