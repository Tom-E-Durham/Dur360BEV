"""
Module: Dual-fisheye BEV segmentation network
Author: Wenke E (wenke.e@durham.ac.uk)
Description: This code is developed based on the SimpleBEV project (https://github.com/aharley/simple_bev/).
Version: 1.0
"""

from torchvision.models.resnet import resnet18, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
import utils.vox
import utils.basic
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")


EPS = 1e-4


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,
                        mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        weights = ResNet101_Weights.DEFAULT
        resnet = resnet101(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)
        return x


class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(weights=None, zero_init_residual=True)
        self.first_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels,
                      kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes,
                      kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            # note [-2] instead of [-3], since Y is gone now
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(
            x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }


def XYZ2xy(X, Y, Z):
    theta = torch.arctan2(Y, Z)
    phi = torch.arctan(torch.sqrt(Y**2 + Z**2) / X)
    FoV = 203 * np.pi / 180
    r = phi * 2 / FoV
    x = r * torch.cos(theta)
    y = abs(r) * torch.sin(theta)
    return x, y


def rotate_y_axis(xyz_fix, angle):
    # Rotation matrix for anti-clockwise rotation around Y-axis
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotation_matrix = torch.tensor([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ], dtype=xyz_fix.dtype, device=xyz_fix.device)

    # Apply the rotation
    xyz_rotated = torch.matmul(xyz_fix, rotation_matrix)
    return xyz_rotated


def unproject_image_to_mem(feat_img, xyz_fix, Z, Y, X):
    B, N, _ = xyz_fix.shape

    # Rotate coordinates
    xyz_fix_rotated = rotate_y_axis(xyz_fix, angle=torch.tensor(np.pi / 2))

    X_w = xyz_fix_rotated[:, :, 0]
    Y_w = xyz_fix_rotated[:, :, 1]
    Z_w = xyz_fix_rotated[:, :, 2]

    x, y = XYZ2xy(X_w, Y_w, Z_w)
    front_mask = X_w > 0

    # Bilinear-Subsample
    x_front = (x+1)/2
    x_back = (x-1)/2
    x_front = x_front * front_mask
    x_back = x_back * ~front_mask

    x = x_front + x_back

    z = torch.zeros_like(x)
    xyz_fix = torch.stack([x, y, z], axis=2)

    xyz_fix = torch.reshape(xyz_fix, [B, Z, Y, X, 3])
    feat_mem = nn.functional.grid_sample(feat_img.unsqueeze(2), xyz_fix,
                                         padding_mode='zeros',
                                         align_corners=False)
    return feat_mem


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid=None):
        loss = self.loss_fn(ypred, ytgt)
        if valid is None:
            loss = utils.basic.reduce_masked_mean(loss, torch.ones_like(loss))
        else:
            loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', trainable=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.trainable = trainable
        if trainable:
            self.gamma_param = nn.Parameter(
                torch.tensor(0, dtype=torch.float32))
        else:
            self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')

        if self.trainable:
            # Apply softplus to ensure gamma is positive
            gamma = F.softplus(self.gamma_param)
        else:
            gamma = self.gamma

        # Get the probabilities of the positive class
        pt = torch.exp(-BCE_loss)

        # Compute the Focal Loss
        F_loss = self.alpha * (1-pt)**gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None,
                 use_lidar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X
        self.use_lidar = use_lidar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        #self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        #self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        self.mean = torch.as_tensor([0.2941, 0.3056, 0.3148]).reshape(
            1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.3561, 0.3668, 0.3749]).reshape(
            1, 3, 1, 1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        if self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3,
                          padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3,
                              padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True)

        if vox_util is not None:
            self.xyz_mem = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_fix = vox_util.Mem2Ref(
                self.xyz_mem, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

    def forward(self, imgs, rad_occ_mem0=None):
        '''
        B = batch size, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        B, C, H, W = imgs.shape
        assert(C == 3)

        # rgb encoder
        device = imgs.device
        imgs = (imgs + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = imgs.shape
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            imgs[self.rgb_flip_index] = torch.flip(
                imgs[self.rgb_flip_index], [-1])

        # get the imagae features using resnet 101
        feat_imgs = self.encoder(imgs)
        if self.rand_flip:
            feat_imgs[self.rgb_flip_index] = torch.flip(
                feat_imgs[self.rgb_flip_index], [-1])

        _, C, Hf, Wf = feat_imgs.shape

        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        xyz_fix = self.xyz_fix.to(feat_imgs.device).repeat(B, 1, 1)
        feat_mem = unproject_image_to_mem(feat_imgs, xyz_fix, Z, Y, X)
        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(
                feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(
                feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(
                    rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(
                    rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        if self.use_lidar:
            assert(rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(
                B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else:  # rgb only
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(
                    B, self.feat2d_dim*Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(
            feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e
