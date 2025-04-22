"""
Module: Project Point Cloud to Image
Author: Wenke E (wenke.e@durham.ac.uk)
Version: 1.0
"""

######################## Import Packages ###################################
import cv2
import matplotlib.pyplot as plt
import sys
import fisheye_tools
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R

######################### Functions #########################################


def rotate_image(image, dim=[640, 1280]):
    h, w = dim
    left = image[:, :h]
    right = image[:, h:]
    new_left = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)
    new_right = cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_image = np.concatenate((new_left, new_right), axis=1)
    return new_image


def c_to_face(cube_image, direction='front_direction', face_w=512):
    if direction == 'front_direction':
        ret = cube_image[face_w:face_w*2, face_w*2:face_w*3]
    elif direction == 'back_direction':
        ret = cube_image[face_w:face_w*2, :face_w]
    elif direction == "left_direction":
        ret = cube_image[face_w:face_w*2, face_w:face_w*2]
    elif direction == "right_direction":
        ret = cube_image[face_w:face_w*2, face_w*3:]
    else:
        print('Error: Incorrect Direction Input.')
    return ret


def project_point_cloud_to_image(point_cloud, intrinsic_matrix, distortion_coeffs, extrinsic_matrix, image, max_depth=5.0, radius=1):
    # Assuming point_cloud is a numpy array of shape (num_points, 4)
    points = point_cloud[:, :3]  # Drop the last column if it's homogeneous
    # Convert matrix to a NumPy array if it's not already
    extrinsic_matrix = np.array(extrinsic_matrix)
    intrinsic_matrix = np.array(intrinsic_matrix)
    distortion_coeffs = np.array(distortion_coeffs)
    # Transform point cloud to camera's coordinate system
    points_cam_coord = points.dot(
        extrinsic_matrix[:3, :3].T) + extrinsic_matrix[:3, 3]

    # Filter points behind the camera
    valid_points_mask = points_cam_coord[:, 2] > 0
    points_cam_coord = points_cam_coord[valid_points_mask]

    # Project to 2D image plane (homogeneous coordinates)
    points_homogeneous = points_cam_coord.dot(intrinsic_matrix.T)

    # Convert from homogeneous to 2D coordinates
    points_2d = points_homogeneous[:, :2] / \
        points_homogeneous[:, 2, np.newaxis]

    # Create an image to store the projection
    undistorted_img = cv2.undistort(image, intrinsic_matrix, distortion_coeffs)

    # Generate a colormap for depth values
    min_depth = np.min(points_cam_coord[:, 2])
    max_depth = np.max(points_cam_coord[:, 2])
    normalized_depths = (
        points_cam_coord[:, 2] - min_depth) / (max_depth - min_depth)

    colormap = plt.get_cmap('hot')
    # Compute RGBA values using the colormap
    rgba_colors = colormap(normalized_depths)

    # Extract RGB values and scale to [0, 255]
    colors = (255 * rgba_colors[:, :3]).astype(np.uint8)

    y_indices = np.clip(points_2d[:, 1].astype(np.int32), 0, image.shape[0]-1)
    x_indices = np.clip(points_2d[:, 0].astype(np.int32), 0, image.shape[1]-1)

    undistorted_img[y_indices, x_indices] = colors

    return undistorted_img


def project(direction, cube_img, points, cali_data):
    face_w = int(cube_img.shape[0]/3)
    image = c_to_face(cube_img, direction=direction, face_w=face_w).copy()

    distortion_coeffs = cali_data[direction]['distortion_coeffs']
    extrinsic = cali_data[direction]["extrinsic"]
    camera_matrix = cali_data[direction]["intrinsic"]

    projection_image = project_point_cloud_to_image(
        points, camera_matrix, distortion_coeffs, extrinsic, image, radius=1)
    return projection_image


def XYZ2uv_poly(X, Y, Z, FOV=203, alpha=1.0):
    EPS = 1e-6
    theta = np.arctan2(Y, Z)  # -pi , pi
    # range [-pi/2 , pi/2] depends on X (front or back)
    phi = np.arctan(np.sqrt(Y**2 + Z**2) / (X+EPS))

    test_mask = phi > 0
    # r = f(phi)

    fov1 = 196/180 * np.pi
    fov2 = FOV/180 * np.pi
    a = 2 / fov1
    c = a * alpha
    b = (1 - c) / (fov2/2 - alpha)
    d = 1 - b * fov2 / 2
    k = 20

    x1 = phi * test_mask
    S = 1 / (1 + np.exp(-k * (x1 - alpha)))
    r1 = (1 - S) * (a * x1) + S * (b * x1 + d)

    x2 = - phi * ~test_mask
    S = 1 / (1 + np.exp(-k * (x2 - alpha)))
    r2 = -1 * ((1 - S) * (a * x2) + S * (b * x2 + d))

    r = r1 + r2
    x = r * np.cos(theta)
    y = abs(r) * np.sin(theta)

    front_mask = X > 0

    x_front = (x+1)/2
    x_back = (x-1)/2
    x_front = x_front * front_mask
    x_back = x_back * ~front_mask

    x = x_front + x_back

    u = (x+1)/2 * 1280
    v = (-y+1)/2 * 640

    u_dist = np.clip(u.astype(np.int32), 0, 1280-1)
    v_dist = np.clip(v.astype(np.int32), 0, 640-1)
    return u_dist, v_dist


def apply_transformation(points, R, T):
    N = points.shape[0]
    ones = np.ones((N, 1))
    homogeneous_points = np.hstack((points, ones))

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    transformed_homogeneous_points = homogeneous_points.dot(
        transformation_matrix.T)

    transformed_points = transformed_homogeneous_points[:, :3]
    return transformed_points


def degree2Rmatrix(x, y, z):
    x_rad = np.radians(x)
    y_rad = np.radians(y)
    z_rad = np.radians(z)

    rotation = R.from_euler('xyz', [x_rad, y_rad, z_rad])
    R_matrix = rotation.as_matrix()
    return R_matrix


def dualfisheye_project(img, src_pcd, R, T, FOV, alpha):
    '''
    pcd: X: forward, Y: left, Z: top
    R: 3x3
    T: 3X1
    '''
    pcd = np.dot(src_pcd,
                 np.array([[1, 0,  0],
                           [0, 0, -1],
                           [0, 1,  0]]))
    pcd = apply_transformation(pcd, R, T)

    X_w = pcd[:, 0]  # forward
    Y_w = pcd[:, 1]  # top
    Z_w = pcd[:, 2]  # right

    u_dist, v_dist = XYZ2uv_poly(X_w, Y_w, Z_w, FOV, alpha)
    # Generate a colormap for depth values
    max_depth = np.max(abs(Z_w))
    normalized_depths = abs(Z_w) / max_depth * 5

    # Compute RGBA values using the colormap
    rgba_colors = plt.cm.viridis(normalized_depths)
    colors = (255 * rgba_colors[:, :3]).astype(np.uint8)

    ret = img.copy()
    ret[v_dist, u_dist] = colors
    return ret

########################## process ####################################################


def initialization(cali_json):

    # Read from a JSON file
    with open(cali_json, 'r') as json_file:
        cali_data = json.load(json_file)

    # get dual fisheye to cubemap projection map
    init_img = np.zeros((640, 1280, 3))
    cubemap_x, cubemap_y = fisheye_tools.getcvmap.dualfisheye2cube(
        init_img, face_w=512)

    return cali_data, cubemap_x, cubemap_y


def dualfisheye_initialization(Tx, Ty, Tz, Rx, Ry, Rz):
    T = np.array([Tx, Ty, Tz]).T
    R = degree2Rmatrix(Rx, Ry, Rz)
    return R, T


def do_projection(pcd, img, cali_data, cubemap_x, cubemap_y):
    # Preprocess image to cubemap
    rot_img = rotate_image(img)
    cube_img = cv2.remap(rot_img, cubemap_x, cubemap_y,
                         interpolation=cv2.INTER_LINEAR)

    # Do Lidar projection on image
    dis_img = None
    for direction in ['left_direction', 'front_direction', 'right_direction', 'back_direction']:
        if dis_img is None:
            dis_img = project(direction, cube_img, pcd, cali_data).copy()
        else:
            dis_img = np.hstack(
                (dis_img, project(direction, cube_img, pcd, cali_data).copy()))
    return dis_img
