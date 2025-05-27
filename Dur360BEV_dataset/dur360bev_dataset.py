import os
import torch
try:
    from Dur360BEV_dataset.utils import load_pcd, map_api, pcd_img
except:
    from utils import load_pcd, map_api, pcd_img
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import fisheyetools as ft
import json
import cv2
import math
import pickle
######################## Dataset ########################
"""
Dur360BEV_Dataset
├── image
│   ├── data
│   │   ├── 0000000000.png
│   │   └── ...
│   └── timestamps.txt
├── labels
│   ├── data
│   │   ├── 0000000000.txt
│   │   └── ...
│   ├── dataformat.txt
│   └── timestamps.txt
├── ouster_points
│   ├── data
│   │   ├── 0000000000.bin
│   │   └── ...
│   ├── dataformat.txt
│   └── timestamps.txt
└── oxts
    ├── data
    │   │   ├── 0000000000.txt
    │   └── ...
    ├── dataformat.txt
    └── timestamps.txt
"""


class Dur360BEV(Dataset):
    def __init__(self,
                 root_dir,
                 img_type='dual_fisheye',
                 map_r=100,
                 map_scale=2,
                 transform=None,
                 offset_orient=False,
                 is_train=True,
                 bev_labels=['Car', 'Pedestrian', 'Lane'],
                 version='initial'):
        """
        Parameters:
            root_dir (string): Directory with all the data.
            img_type (string): The type of image to load. Options: 'dual_fisheye', 'equi_img'
            map_r (int): The range of the map in meters.
            map_scale (int): The number of pixels per meter
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): If True, the dataset is for training. If False, the dataset is for visulising.
            labels (list): The list of labels to include in the BEV map. Options: 'Car', 'Pedestrian', 'Lane'
        """
        # Directories and flags
        self.root_dir = root_dir

        assert img_type in ['dual_fisheye',
                            'equi_img'], f"Invalid image type: {img_type}"
        self.img_type = img_type

        # Define dataset directories
        self.img_dir = os.path.join(root_dir, "image/data")
        self.label_dir = os.path.join(root_dir, "labels/data")
        self.pcd_dir = os.path.join(root_dir, "ouster_points/data")
        self.gps_dir = os.path.join(root_dir, "oxts/data")

        # Load timestamps for consistency
        self.img_timestamps = self._load_timestamps(
            os.path.join(root_dir, "image/timestamps.txt"))
        self.label_timestamps = self._load_timestamps(
            os.path.join(root_dir, "labels/timestamps.txt"))
        self.pcd_timestamps = self._load_timestamps(
            os.path.join(root_dir, "ouster_points/timestamps.txt"))
        self.gps_timestamps = self._load_timestamps(
            os.path.join(root_dir, "oxts/timestamps.txt"))
        
        self.filenames = sorted([
            f[:10] for f in os.listdir(self.img_dir) if f.endswith('.png') and len(f) == 14
        ])
        # Ensure all timestamps match
        assert len(self.img_timestamps) == len(self.label_timestamps) == len(self.pcd_timestamps) == len(self.gps_timestamps), \
            "Mismatch in dataset timestamps!"

        self.transform = transform
        self.is_train = is_train

        self.map_r = map_r  # map range in meters
        self.map_scale = map_scale  # pixels per meter
        self.offset_orient = offset_orient

        if 'Lane' in bev_labels:
            self.bin_map = True
        else:
            self.bin_map = False
        self.bev_labels = bev_labels
        if self.bin_map:
            Map = map_api.OSMSemanticMap()
            self.Vis = map_api.OSMSemanticMapVis(Map)

        # PointCloud loader
        self.pcd_loader = load_pcd.PointCloud_Loader()

        # Lidar image loader
        pcd_meta_dir = os.path.join(self.root_dir, "metadata/os1.json")
        self.pcd_img_loader = pcd_img.GetLidarImages(pcd_meta_dir)

    def _load_timestamps(self, timestamp_file):
        """Load timestamps from a text file."""
        with open(timestamp_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.img_timestamps)

    def pre_dualfisheye(self, img):
        """
        Pre-process dual-fisheye images:
            Rotate left and right fisheye
        """
        h, w = img.shape[:2]
        if h*2 != w:
            h = w // 2
            img = img[:h]
        img_l = cv2.rotate(img[:, :h], cv2.ROTATE_90_CLOCKWISE)
        img_r = cv2.rotate(img[:, h:], cv2.ROTATE_90_COUNTERCLOCKWISE)

        return np.concatenate((img_l, img_r), axis=1)

    def get_image_data(self, idx):
        """
        Load image data with cv2 and change color to RGB before 
        pre-processing the dual-fisheye image
        """
        # img_path = os.path.join(self.img_dir, f"{str(idx).zfill(10)}.png")
        img_path = os.path.join(self.img_dir, f"{self.filenames[idx]}.png")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.pre_dualfisheye(image)

        if self.img_type == 'equi_img':
            if not hasattr(self, 'maps'):
                self.maps = ft.getcvmap.dualfisheye2equi(image,
                                                         size=(2048, 1024),
                                                         aperture=203,
                                                         center_angle=0)

            image = cv2.remap(
                image, self.maps[0], self.maps[1], interpolation=cv2.INTER_CUBIC)

        if self.transform:
            image = self.transform(image)
        return image

    def get_pcd_data(self, idx):
        """
        Load the point cloud data with 9 field:
        x, y, z, intensity, time, reflectivity, ring, ambient, range
        """
        """Load point cloud data from .bin file."""
        # pcd_path = os.path.join(self.pcd_dir, f"{str(idx).zfill(10)}.bin")
        pcd_path = os.path.join(self.pcd_dir, f"{self.filenames[idx]}.bin")
        # KITTI-style format
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 9)
        return torch.tensor(pcd)

    def process_lidar_image(self, image, target_size=(2048, 242), if_padding=False):
        """
        Process the lidar images to the equi-rectangular format
        """
        H, W = image.shape
        assert H == 128 and W == 2048, f"Invalid image shape: {H}x{W}"
        resized_img = cv2.resize(image.numpy(), target_size)
        if if_padding:
            padding = (391, 2048)
            lidar_image = np.pad(
                resized_img, ((padding[0], padding[0]), (0, 0)), mode='constant')
        else:
            lidar_image = resized_img
        return torch.tensor(lidar_image).unsqueeze(0)

    def get_pcd_imgs(self, pcd, equi_format=False):
        """
        Load and process binary lidar images:
            range_img
            ambient_img
            intensity_img
            reflectivity_img
        """
        lidar_imgs = self.pcd_img_loader(pcd)
        range_img = lidar_imgs['range_img']
        ambient_img = lidar_imgs['ambient_img']
        intensity_img = lidar_imgs['intensity_img']
        reflectivity_img = lidar_imgs['reflectivity_img']

        if not equi_format:
            return {'range_img': range_img.unsqueeze(0),
                    'ambient_img': ambient_img.unsqueeze(0),
                    'intensity_img': intensity_img.unsqueeze(0),
                    'reflectivity_img': reflectivity_img.unsqueeze(0)}

        else:
            range_img = self.process_lidar_image(range_img)
            ambient_img = self.process_lidar_image(ambient_img)
            intensity_img = self.process_lidar_image(intensity_img)
            reflectivity_img = self.process_lidar_image(reflectivity_img)
            return {'range_img': range_img,
                    'ambient_img': ambient_img,
                    'intensity_img': intensity_img,
                    'reflectivity_img': reflectivity_img}

    def rotate_point(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def draw_rotated_rectangle(self, img, center, width, height, angle, seg_idx):
        """
        Draw a rectangle with a given rotation.

        :param img: The image to draw on.
        :param center: A tuple (x, y) for the center of the rectangle.
        :param width: The width of the rectangle.
        :param height: The height of the rectangle.
        :param angle: The rotation angle in degrees. Positive angles rotate counter-clockwise.
        """
        angle_rad = math.radians(angle)
        half_width, half_height = width / 2, height / 2
        corners = [
            (center[0] - half_width, center[1] - half_height),
            (center[0] + half_width, center[1] - half_height),
            (center[0] + half_width, center[1] + half_height),
            (center[0] - half_width, center[1] + half_height)
        ]

        # Rotate the corners and convert them to integer coordinates
        rotated_corners = np.array(
            [self.rotate_point(center, pt, angle_rad) for pt in corners], np.int32)
        rotated_corners = rotated_corners.reshape((-1, 1, 2))

        # Draw the filled rotated rectangle
        cv2.fillPoly(img, [rotated_corners], seg_idx)

    def get_rotation_matrix(self, yaw):
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rotation_matrix = torch.stack([
            # First row: [cos(yaw), sin(yaw)]
            torch.stack([cos_yaw, sin_yaw], dim=-1),
            # Second row: [-sin(yaw), cos(yaw)]
            torch.stack([-sin_yaw, cos_yaw], dim=-1)
        ], dim=-2)
        return rotation_matrix

    def get_label(self, idx):
        """Load 3D bounding box annotations from .txt file."""
        # label_path = os.path.join(self.label_dir, f"{str(idx).zfill(10)}.txt")
        label_path = os.path.join(self.label_dir, f"{self.filenames[idx]}.txt")
        objects = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                obj_type = data[0]
                dz, dy, dx, x, y, z, yaw = map(float, data[1:])
                if obj_type in ['Car', 'Bus', 'Truck'] and 'Car' in self.bev_labels:
                    labels.append(torch.tensor(0.0))  # Car Label
                    objects.append(torch.Tensor([x, y, z, dx, dy, dz, yaw]))
                elif obj_type == 'Pedestrian' and 'Pedestrian' in self.bev_labels:
                    labels.append(torch.tensor(1.0))  # Pedestrian Label
                    objects.append(torch.Tensor([x, y, z, dx, dy, dz, yaw]))

        if len(objects):
            objects = torch.stack(objects, dim=0)
            labels = torch.stack(labels, dim=0)
        else:
            objects = torch.zeros((0, 9))
            labels = torch.zeros(0)

        return objects, labels

    def get_gps_imu(self, idx):
        """Load GPS and IMU data from .txt file."""
        # gps_path = os.path.join(self.gps_dir, f"{str(idx).zfill(10)}.txt")
        gps_path = os.path.join(self.gps_dir, f"{self.filenames[idx]}.txt")
        with open(gps_path, 'r') as f:
            lat, lon, alt, roll, pitch, yaw = map(
                float, f.readline().strip().split())

        return torch.tensor([lat, lon, alt]), torch.tensor([roll, pitch, yaw])

    def get_anno_lists(self, idx):
        """
        Load Point Cloud 3D bounding box annotations. 
        """

        if not os.path.exists(self.pcd_anno_dir):
            pcd_anno = None
            pass
        else:
            #print('Annotation Found.')
            pcd_anno_name = os.path.join(self.pcd_anno_dir, sorted(
                os.listdir(self.pcd_anno_dir))[idx])

            with open(pcd_anno_name, 'r') as file:
                pcd_annos = json.load(file)

        obj_list = []  # store object center,rot,size
        anno_list = []  # store object label 0: car, 1: pedestrian
        for pcd_anno in pcd_annos:
            version = pcd_anno['version']
            if version == '1.0':
                objects = pcd_anno['instances']
            else:
                objects = pcd_anno['objects']

            for object in objects:
                if object['className']:
                    label = object['className']  # from munual label
                else:
                    label = object['modelClass']  # from model
                contour = object["contour"]
                center3D = contour['center3D']
                c_x, c_y, c_z = center3D.values()  # x : forward, y : left, z : up
                # ignore the ergo car
                if (abs(c_x) < 2.5) and (abs(c_y) < 0.9) and (label in ['Car', 'Bus', 'Truck']):
                    pass
                else:
                    rotation3D = contour['rotation3D']
                    r_x, r_y, r_z = rotation3D.values()
                    size3D = contour['size3D']
                    s_x, s_y, s_z = size3D.values()

                    crs = np.array([c_x, c_y, c_z, r_x, r_y, r_z,
                                    s_x, s_y, s_z]).astype(np.float32)
                    crs = torch.Tensor(crs)
                    if label == 'Car' or label == 'Bus' or label == 'Truck':
                        if 'Car' in self.bev_labels:
                            anno_list.append(torch.tensor(0.0))  # Car Label
                            obj_list.append(crs)
                    elif label == 'Pedestrian':
                        if 'Pedestrian' in self.bev_labels:
                            # Pedestrian Label
                            anno_list.append(torch.tensor(1.0))
                            obj_list.append(crs)
        if len(obj_list):
            obj_list = torch.stack(obj_list, dim=0)
            anno_list = torch.stack(anno_list, dim=0)
        else:
            obj_list = torch.zeros((0, 9))
            anno_list = torch.zeros(0)

        return obj_list, anno_list

    def get_bev_center_offset(self, obj_list, radius, map_r, scale):
        """
        Load ground truth BEV centerness and offset images

        Directions:
            x: forward
            y: left
        """
        map_res = int(map_r * scale)
        x = obj_list[:, 0] * -scale + map_res / 2
        y = obj_list[:, 1] * -scale + map_res / 2
        yaw = obj_list[:, 5]
        rotation_matrices = self.get_rotation_matrix(yaw)
        xy = torch.stack([x, y], dim=1)  # N,2
        N, _ = xy.shape
        if N:
            grid_v = torch.linspace(0.0, map_res-1.0, map_res)
            grid_v = torch.reshape(grid_v, [1, map_res])
            grid_v = grid_v.repeat(map_res, 1)
            grid_u = torch.linspace(0.0, map_res-1.0, map_res)
            grid_u = torch.reshape(grid_u, [map_res, 1])
            grid_u = grid_u.repeat(1, map_res)

            # get mesh
            grid = torch.stack([grid_u, grid_v], dim=0)  # 2,200,200

            xy = xy.reshape(N, 2, 1, 1)
            grid = grid.reshape(1, 2, map_res, map_res)
            xy = xy.round()

            off = xy - grid
            radius = 3
            # Compute Distance Grid
            dist_grid = torch.sum(off**2, dim=1, keepdim=False)
            # Compute Gaussian Mask
            mask = torch.exp(-dist_grid/(2*radius*radius))
            mask[mask < 0.001] = 0.0

            center = torch.max(mask, dim=0, keepdim=True)[0]

            if self.offset_orient:
                #### Add orientation features to offset map ####
                # Reshape off to match the dimensions required for batch matrix multiplication
                # Shape: (N, 2, 200*200)
                off_reshaped = off.reshape(off.shape[0], 2, -1)
                # Apply the rotation matrices using batch matrix multiplication
                rotated_off_reshaped = torch.bmm(
                    rotation_matrices, off_reshaped)  # Shape: (N, 2, 200*200)
                # Reshape back to the original shape
                off = rotated_off_reshaped.reshape(
                    off.shape)  # Shape: (N, 2, 200, 200)

        else:
            center = torch.zeros((1, map_res, map_res))
            off = torch.zeros((N, 2, map_res, map_res))

        return center, off

    def get_bev_seg(self, obj_list, anno_list, map_r=100, scale=2):
        """
        Load ground truth BEV segmentation maps.

        Parameters:
            map_r: meters in x and y direction
            scale: pixels per meter
        Directions:
            x : forward, y : left, z : up, in PCD space
        """
        N, _ = obj_list.shape

        n_seg = int(len(self.bev_labels))
        seg = np.zeros((n_seg, map_r * scale, map_r * scale))

        radius = 3
        center, offset = self.get_bev_center_offset(
            obj_list, radius, map_r, scale)
        assert (offset.shape[0] == N)
        masklist = torch.zeros(
            (N, 1, map_r * scale, map_r * scale), dtype=torch.float32)
        # draw bev seg map
        for n in range(N):
            # dx, dy, dz, x, y, z, yaw
            c_x, c_y, c_z, s_x, s_y, s_z, r_z = obj_list[n]
            label = int(anno_list[n])

            u = int(-c_y*scale+seg.shape[1]/2)
            v = int(-c_x*scale+seg.shape[1]/2)
            uv = (u, v)
            # Replace with the rotation angle in degrees
            angle = math.degrees(-r_z)+90
            self.draw_rotated_rectangle(
                seg[label], uv, s_x*scale, s_y*scale, angle, n+1.0)
            # 1, map_r * scale, map_r * scale
            inst = (seg[label] == (n+1.0)).astype(np.float32)
            masklist[n, 0] = torch.tensor((inst > 0.01), dtype=torch.float32)

        offset = offset * masklist
        offset = torch.sum(offset, dim=0)

        return torch.Tensor(seg), center, offset

    def get_bin_map(self, gps_location, yaw=None):
        """
        Load binary map tiles from OSM using gps.
        """
        bin_map = self.Vis.get_local_bin(gps_location,
                                         yaw,
                                         search_range=self.map_r,
                                         map_scale=self.map_scale)
        return torch.Tensor(bin_map)

    def render_colored_bev(self, bev_seg, ego_car_size=(1.2, 2.5)):
        """
        Generates a colored Bird's Eye View (BEV) image.

        Parameters:
            bev_seg (numpy.ndarray): A (3, H, W) array where:
                - Channel 0: Cars
                - Channel 1: Pedestrians
                - Channel 2: Map tiles
            ego_car_size (tuple): (length, width) of the ego car in meters.
            scale (float): Pixels per meter.
        Returns:
            bev_colored (numpy.ndarray): A (H, W, 3) RGB image.
        """

        H, W = bev_seg.shape[1], bev_seg.shape[2]

        # Initialize an RGB image (H, W, 3) with a white background
        bev_colored = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Define colors (BGR format for OpenCV)
        color_map_tile = (180, 220, 255)  # Light Orange (BGR)
        color_car = (255, 0, 0)  # Blue (BGR)
        color_pedestrian = (0, 0, 255)  # Red (BGR)
        color_ego = (150, 80, 220)  # Beautiful Purple (BGR)

        # Draw map tiles first (background)
        if 'Lane' in self.bev_labels:
            bev_colored[bev_seg[2] > 0] = color_map_tile

        # Draw cars (on top of map tiles)
        if 'Car' in self.bev_labels:
            bev_colored[bev_seg[0] > 0] = color_car

        # Draw pedestrians (on top of everything)
        if 'Pedestrian' in self.bev_labels:
            bev_colored[bev_seg[1] > 0] = color_pedestrian

        # Draw the ego car in the center
        ego_x, ego_y = W // 2, H // 2  # Center of the image
        ego_w, ego_h = int(ego_car_size[0] * self.map_scale), int(
            ego_car_size[1] * self.map_scale)  # Convert size to pixels
        top_left = (ego_x - ego_w // 2, ego_y - ego_h // 2)
        bottom_right = (ego_x + ego_w // 2, ego_y + ego_h // 2)
        cv2.rectangle(bev_colored, top_left, bottom_right,
                      color_ego, thickness=-1)  # Filled rectangle

        # Convert BGR to RGB for correct visualization
        bev_colored = cv2.cvtColor(bev_colored, cv2.COLOR_BGR2RGB)

        return bev_colored

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
        r = torch.where((0 <= h) & (h < 1/6), c, torch.where((1/6 <= h) & (h < 2/6), x, torch.where((2/6 <= h) & (
            h < 3/6), zeros, torch.where((3/6 <= h) & (h < 4/6), zeros, torch.where((4/6 <= h) & (h < 5/6), x, c)))))
        g = torch.where((0 <= h) & (h < 1/6), x, torch.where((1/6 <= h) & (h < 2/6), c, torch.where((2/6 <= h) & (
            h < 3/6), c, torch.where((3/6 <= h) & (h < 4/6), x, torch.where((4/6 <= h) & (h < 5/6), zeros, zeros)))))
        b = torch.where((0 <= h) & (h < 1/6), zeros, torch.where((1/6 <= h) & (h < 2/6), zeros, torch.where(
            (2/6 <= h) & (h < 3/6), x, torch.where((3/6 <= h) & (h < 4/6), c, torch.where((4/6 <= h) & (h < 5/6), c, x)))))

        rgb = torch.stack([r, g, b], dim=1) + \
            m.unsqueeze(1)  # Shape: (B, 3, H, W)

        return rgb

    def render_colored_offset(self, offset_map, clip=5.0):
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
        magnitude = torch.sqrt(
            offset_map[:, 0] ** 2 + offset_map[:, 1] ** 2)  # Shape: (B, H, W)

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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Image
        image = self.get_image_data(idx)

        # Pointcloud
        points = self.get_pcd_data(idx)

        # Lidar images
        lidar_imgs = self.get_pcd_imgs(points, equi_format=True)

        # PCD Annotation
        obj_list, anno_list = self.get_label(idx)
        bev_seg, center, offset = self.get_bev_seg(
            obj_list, anno_list, map_r=self.map_r, scale=self.map_scale)
        bev_seg = (bev_seg > 0).float()

        # GPS
        gps, odom = self.get_gps_imu(idx)

        # Binary map tile
        if self.bin_map:
            bin_map = self.get_bin_map([gps[1], gps[0]], float(odom[2]))
            bin_map = bin_map.unsqueeze(0)
            # concatenate the bin_map and the bev_seg
            bev_seg[-1] = bin_map

        sample = {'image': image, 'pcd': points, 'lidar_images': lidar_imgs, 'bev_seg': bev_seg, 'center': center,
                  'offset': offset, 'obj_list': obj_list, 'anno_list': anno_list, 'gps': gps, 'odom': odom}  # , 'imu':imu_data
        #print(f'[DEBUG]: Index: {idx}, image: {image.shape}, bev_seg: {bev_seg.shape}, center: {center.shape}, offset: {offset.shape}')
        if self.is_train:
            sample = {'image': image, 'pcd': points, 'lidar_images': lidar_imgs,
                      'bev_seg': bev_seg, 'center': center, 'offset': offset}
        return sample

#########################################################


#########################################################
def compile_data(root_dir, batch_size, num_workers, img_type='dual_fisheye', map_r=100, map_scale=2, do_shuffle=True, is_train=True, random_split=False):
    if os.path.exists(root_dir):
        totorch_img = transforms.Compose((
            transforms.ToTensor(),
        ))
        dataset = Dur360BEV(root_dir,
                            img_type=img_type,
                            map_r=map_r,
                            map_scale=map_scale,
                            transform=totorch_img,
                            is_train=is_train,
                            bev_labels=['Car'])
        # Split the dataset
        if random_split:
            pkl_path = os.path.join(root_dir, 'dataset_indices_random.pkl')
            with open(pkl_path, 'rb') as f:
                indices = pickle.load(f)
                train_indices = indices['train_indices']
                validation_indices = indices['val_indices']
                test_indices = indices['test_indices']
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, validation_indices)
            test_dataset = Subset(dataset, test_indices)

            # Print the lengths of each dataset
            print(f"Training dataset length: {len(train_dataset)}")
            print(f"Validation dataset length: {len(val_dataset)}")
            print(f"Test dataset length: {len(test_dataset)}")
        else:
            pkl_path = os.path.join(root_dir, 'metadata/dataset_indices.pkl')
            with open(pkl_path, 'rb') as f:
                indices = pickle.load(f)
                train_indices = indices['train_indices']
                test_indices = indices['test_indices']
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)

            # Print the lengths of each dataset
            print(f"Training dataset length: {len(train_dataset)}")
            print(f"Test dataset length: {len(test_dataset)}")

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=do_shuffle,
                                                   num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=do_shuffle,
                                                  num_workers=num_workers)
        if random_split:
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=do_shuffle,
                                                     num_workers=num_workers)

            return train_loader, val_loader, test_loader
        else:
            return train_loader, test_loader
    else:
        print('Compile dataloader failed: path not found')


def prepare_dataset(root_dir, is_train=True, random_split=False):
    totorch_img = transforms.Compose((
        transforms.ToTensor(),
    ))
    dataset = Dur360BEV(root_dir, transform=totorch_img, is_train=is_train)
    # Split the dataset
    if random_split:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dur360_utils/dataset_indices.pkl'), 'rb') as f:
            indices = pickle.load(f)
            train_indices = indices['train_indices']
            val_indices = indices['val_indices']
            test_indices = indices['test_indices']
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Print the lengths of each dataset
        print(f"Training dataset length: {len(train_dataset)}")
        print(f"Validation dataset length: {len(val_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    else:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dur360_utils/dataset_indices.pkl'), 'rb') as f:
            indices = pickle.load(f)
            train_indices = indices['train_indices']
            test_indices = indices['test_indices']
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        # Print the lengths of each dataset
        print(f"Training dataset length: {len(train_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")
        return train_dataset, test_dataset
