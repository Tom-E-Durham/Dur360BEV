"""
This code is for generating and processing the Ouster OS1 Lidar images 
with better visualisation.

The Lidar images include types:
    Range image
    Ambient image
    Reflectivity image
    Intensity image
"""

import numpy as np
import json
import torch


class AutoExposure:
    def __init__(self):
        self.lo_state = -1.0
        self.hi_state = -1.0
        self.lo = -1.0
        self.hi = -1.0
        self.initialized = False
        self.counter = 0

        # Configuration constants
        self.ae_percentile = 0.1
        self.ae_damping = 0.90
        self.ae_update_every = 3
        self.ae_stride = 4
        self.ae_min_nonzero_points = 100

    def __call__(self, image):
        """
        Scales the image so that contrast is stretched between 0 and 1.
        The top percentile becomes 1, and the bottom percentile becomes 0.
        """
        key_array = image.flatten()
        dist_img = image.copy()
        if self.counter == 0:
            nonzero_indices = np.nonzero(key_array[::self.ae_stride])[0]
            nonzero_values = key_array[::self.ae_stride][nonzero_indices]

            if len(nonzero_values) < self.ae_min_nonzero_points:
                # Too few nonzero values; do nothing
                return

            # Percentile calculations
            lo = np.percentile(nonzero_values, self.ae_percentile * 100)
            hi = np.percentile(nonzero_values, (1 - self.ae_percentile) * 100)

            if not self.initialized:
                self.initialized = True
                self.lo_state = lo
                self.hi_state = hi

            self.lo = lo
            self.hi = hi

        if not self.initialized:
            return

        # Apply exponential smoothing
        self.lo_state = self.ae_damping * self.lo_state + \
            (1.0 - self.ae_damping) * self.lo
        self.hi_state = self.ae_damping * self.hi_state + \
            (1.0 - self.ae_damping) * self.hi
        self.counter = (self.counter + 1) % self.ae_update_every

        # Normalize the image
        dist_img -= self.lo_state
        dist_img *= (1.0 - 2 * self.ae_percentile) / \
            (self.hi_state - self.lo_state)

        # Clamp to [0, 1]
        np.clip(dist_img, 0.0, 1.0, out=dist_img)

        return dist_img


class BeamUniformityCorrector:
    def __init__(self):
        self.counter = 0
        self.dark_count = []

        # Configuration constants
        self.buc_damping = 0.92
        self.buc_update_every = 8

    def compute_dark_count(self, image):
        """
        Computes the dark count as the median of row differences.
        """
        image_h, image_w = image.shape
        new_dark_count = np.zeros(image_h)

        row_diffs = image[1:, :] - image[:-1, :]

        for i in range(1, image_h):
            new_dark_count[i] = new_dark_count[i - 1] + \
                np.median(row_diffs[i - 1, :])

        # Remove gradients using a linear fit
        A = np.vstack([np.ones(image_h), np.arange(image_h)]).T
        coeffs = np.linalg.lstsq(A, new_dark_count, rcond=None)[0]
        linear_fit = A @ coeffs
        new_dark_count -= linear_fit

        # Subtract the minimum value to ensure non-negative offsets
        new_dark_count -= new_dark_count.min()

        return new_dark_count

    def update_dark_count(self, image):
        """
        Updates the dark count using exponential smoothing.
        """
        new_dark_count = self.compute_dark_count(image)

        if len(self.dark_count) == 0:
            self.dark_count = new_dark_count
        else:
            self.dark_count = self.buc_damping * np.array(self.dark_count) + \
                (1.0 - self.buc_damping) * new_dark_count

    def __call__(self, image):
        """
        Applies dark count correction to reduce horizontal line artifacts.
        """
        image_h, _ = image.shape
        dist_img = image.copy()

        if self.counter == 0:
            if len(self.dark_count) == 0:
                self.dark_count = self.compute_dark_count(dist_img)
            else:
                self.update_dark_count(dist_img)
        self.counter = (self.counter + 1) % self.buc_update_every

        # Apply dark count correction row by row
        for i in range(image_h):
            dist_img[i, :] -= self.dark_count[i]
            dist_img[i, :] = np.clip(
                dist_img[i, :], 0, np.iinfo(np.uint32).max)

        return dist_img


class GetLidarImages:
    def __init__(self, meta_dir):
        # load meta data
        with open(meta_dir) as f:
            meta_data = json.load(f)
        self.H = meta_data['data_format']['pixels_per_column']
        self.W = meta_data['data_format']['columns_per_frame']
        self.px_offset = np.array(
            meta_data['data_format']['pixel_shift_by_row'])

        # Load image processing tools
        self.ambient_ae = AutoExposure()
        self.intensity_ae = AutoExposure()
        self.reflectivity_ae = AutoExposure()
        self.ambient_buc = BeamUniformityCorrector()

    def __call__(self, pcd):
        # Reshape point cloud into a 2D grid (H x W)
        pcd_np = pcd.numpy().reshape(self.H, self.W, -1)

        # Compute column indices with pixel offset (destaggering)
        column_indices = (np.arange(self.W) + self.W -
                          self.px_offset[:, None]) % self.W

        # Gather data using advanced indexing
        pcd_destaggered = pcd_np[np.arange(self.H)[:, None], column_indices]

        # Extract images
        range_img = pcd_destaggered[:, :, -1] / 1000  # millimeter to metter
        ambient_img = pcd_destaggered[:, :, -2]
        intensity_img = pcd_destaggered[:, :, 3]
        reflectivity_img = pcd_destaggered[:, :, 5]

        # Image processing
        range_img = range_img / range_img.max()  # normalise this to (0,1)
        ambient_img = self.ambient_buc(ambient_img)
        ambient_img = self.ambient_ae(ambient_img)
        intensity_img = self.intensity_ae(intensity_img)
        reflectivity_img = self.reflectivity_ae(reflectivity_img)
        ambient_img = np.sqrt(ambient_img)
        intensity_img = np.sqrt(intensity_img)

        return {"range_img": torch.Tensor(range_img),
                "ambient_img": torch.Tensor(ambient_img),
                "intensity_img": torch.Tensor(intensity_img),
                "reflectivity_img": torch.Tensor(reflectivity_img)}
