"""
Module: TensorBoard Logger
Author: Wenke E (wenke.e@durham.ac.uk)
Version: 1.0
"""

import numpy as np
import torch


class TensorBoardLogger:
    def __init__(self, writer, global_step, log_freq=10, img_freq=100):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = int(log_freq)
        self.img_freq = int(img_freq)
        self.log_save = (self.global_step % self.log_freq == 0)
        self.img_save = (self.global_step % self.img_freq == 0)

    def scalar(self, tag, value):
        """
        Log scalar values: loss, iou, time...
        """
        if self.log_save:
            self.writer.add_scalar(tag, value, self.global_step)

    def rgb_img(self, tag, images):
        """
        Log RGB images
        """
        if self.img_save:
            B, C, H, W = images.shape
            assert (C == 3)
            self.writer.add_image(
                tag, images[0], self.global_step, dataformats='CHW')

    def bin_img(self, tag, images):
        """
        Log binaryt images
        """
        if self.img_save:
            B, H, W = images.shape
            self.writer.add_image(
                tag, images[0], self.global_step, dataformats='HW')

    def offset2color(self, offset_map, clip=5.0):
        """
        Converts a batch of 2-channel offset maps (B, 2, H, W) into RGB images (B, 3, H, W)
        using an HSV-based optical flow coloring approach.

        Args:
            offset_map (torch.Tensor): Shape (B, 2, H, W), where:
                - Channel 0: x-offset
                - Channel 1: y-offset
            clip (float): Maximum absolute value to clamp offsets. Default is 5.

        Returns:
            torch.Tensor: RGB images of shape (B, 3, H, W), normalized between 0 and 1.
        """
        B, C, H, W = offset_map.shape
        assert C == 2, "Offset map should have 2 channels (x, y)."

        # Clone to prevent modifying original tensor
        offset_map = offset_map.clone().detach()

        # Compute magnitude (radius) of the offsets
        # magnitude = torch.sqrt(
        # offset_map[:, 0] ** 2 + offset_map[:, 1] ** 2)  # Shape: (B, H, W)

        # Compute mean and standard deviation for normalization
        abs_image = torch.abs(offset_map)
        mean_offset = abs_image.mean(dim=[2, 3], keepdim=True)  # (B, 2, 1, 1)
        std_offset = abs_image.std(dim=[2, 3], keepdim=True)  # (B, 2, 1, 1)

        # Clip or normalize the offsets
        if clip:
            offset_map = torch.clamp(offset_map, -clip, clip) / clip
        else:
            max_offset = mean_offset + std_offset * 2 + 1e-10
            offset_map = offset_map / max_offset.clamp(min=1)

        # Compute radius (magnitude) and clamp it
        # Shape: (B, H, W)
        radius = torch.sqrt(offset_map[:, 0] ** 2 + offset_map[:, 1] ** 2)
        # Ensure it's within [0,1]
        radius_clipped = torch.clamp(radius, 0.0, 1.0)

        # Compute direction (angle) for color encoding
        # Shape: (B, H, W)
        angle = torch.atan2(offset_map[:, 1], offset_map[:, 0]) / np.pi

        # Convert to HSV format
        # [-π, π] → [0,1] hue color mapping
        hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
        # Fixed saturation for strong colors
        saturation = torch.ones_like(hue) * 0.75
        value = radius_clipped  # Brightness depends on magnitude

        hsv = torch.stack([hue, saturation, value],
                          dim=1)  # Shape: (B, 3, H, W)

        # Convert HSV to RGB
        rgb_image = self.hsv_to_rgb(hsv)  # (B, 3, H, W)

        return rgb_image

    def hsv_to_rgb(self, hsv):
        """
        Converts a batch of HSV images to RGB.

        Args:
            hsv (torch.Tensor): Shape (B, 3, H, W), values in [0,1].

        Returns:
            torch.Tensor: RGB images of shape (B, 3, H, W), values in [0,1].
        """
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]  # Extract channels

        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c

        zeros = torch.zeros_like(h)
        r = torch.where((0 <= h) & (h < 1 / 6), c, torch.where((1 / 6 <= h) & (h < 2 / 6), x, torch.where((2 / 6 <= h) & (
            h < 3 / 6), zeros, torch.where((3 / 6 <= h) & (h < 4 / 6), zeros, torch.where((4 / 6 <= h) & (h < 5 / 6), x, c)))))
        g = torch.where((0 <= h) & (h < 1 / 6), x, torch.where((1 / 6 <= h) & (h < 2 / 6), c, torch.where((2 / 6 <= h) & (
            h < 3 / 6), c, torch.where((3 / 6 <= h) & (h < 4 / 6), x, torch.where((4 / 6 <= h) & (h < 5 / 6), zeros, zeros)))))
        b = torch.where((0 <= h) & (h < 1 / 6), zeros, torch.where((1 / 6 <= h) & (h < 2 / 6), zeros, torch.where(
            (2 / 6 <= h) & (h < 3 / 6), x, torch.where((3 / 6 <= h) & (h < 4 / 6), c, torch.where((4 / 6 <= h) & (h < 5 / 6), c, x)))))

        rgb = torch.stack([r, g, b], dim=1) + \
            m.unsqueeze(1)  # Shape: (B, 3, H, W)

        return rgb
