import numpy as np
from . import e2c


def lerp(y0, y1, x0, x1, x):
    m = (y1 - y0) / (x1 - x0)
    b = y0
    return m * (x-x0) + b


def equicoortofisheyecoor(x, y, aperture):
    # From equirectangular coordinates (x,y) to fisheye coordinates (r,theta)
    # 计算极坐标参数
    longitude = x * np.pi
    latitude = y * np.pi / 2

    # 3D XYZ
    p_x = np.cos(latitude) * np.cos(longitude)
    p_y = np.cos(latitude) * np.sin(longitude)
    p_z = np.sin(latitude)
    p_xz = np.sqrt(p_x ** 2 + p_z ** 2)

    # 计算源图像坐标
    r = 2 * np.arctan2(p_xz, p_y) / aperture
    theta = np.arctan2(p_z, p_x)

    return r, theta


def dualfisheye2equi(frame, size=[2000, 1000], aperture=203, center_angle=0):
    # check center angle
    if not (0 <= center_angle < 360):
        raise ValueError(
            f"Parameter 'center_angle' out of range: [0, 360). Got {center_angle}.")

    aperture = aperture * np.pi / 180
    h_src, w_src = frame.shape[:2]
    w_dst, h_dst = size

    # Create mesh grid coordinates
    x_dst_norm_L, y_dst_norm_L = np.meshgrid(np.linspace(1, -1, w_dst),
                                             np.linspace(-1, 1, h_dst))
    x_dst_norm_R, y_dst_norm_R = np.meshgrid(np.linspace(-1, 1, w_dst),
                                             np.linspace(-1, 1, h_dst))

    r_L, theta_L = equicoortofisheyecoor(x_dst_norm_L, y_dst_norm_L, aperture)
    r_R, theta_R = equicoortofisheyecoor(x_dst_norm_R, y_dst_norm_R, aperture)
    x_src_norm_L = r_L * np.cos(theta_L)+1
    x_src_norm_R = r_R * np.cos(theta_R)-1
    y_src_norm_L = r_L * np.sin(theta_L)
    y_src_norm_R = r_R * np.sin(theta_R)

    x_src_L = lerp(0, w_src/2, 0, 2, x_src_norm_L)
    x_src_R = lerp(w_src/2, w_src, 0, -2, x_src_norm_R)
    y_src_L = lerp(0, h_src, -1, 1, y_src_norm_L)
    y_src_R = lerp(0, h_src, -1, 1, y_src_norm_R)

    # supppres out of the bound index error (warning this will overwrite multiply pixels!)
    x_src_L_ = np.minimum(w_src - 1, np.floor(x_src_L).astype(int))
    x_src_R_ = np.minimum(w_src - 1, np.floor(x_src_R).astype(int))
    x_src_ = np.concatenate(
        (x_src_L_[:, :int(w_dst/2)], x_src_R_[:, int(w_dst/2):]), axis=1)

    y_src_L_ = np.minimum(h_src - 1, np.floor(y_src_L).astype(int))
    y_src_R_ = np.minimum(h_src - 1, np.floor(y_src_R).astype(int))
    y_src_ = np.concatenate(
        (y_src_L_[:, :int(w_dst/2)], y_src_R_[:, int(w_dst/2):]), axis=1)

    # shift the angle to align with the middle of the image (90 degree)
    adjusted_angle = (center_angle - 90) % 360
    # calculate offset to desired center angle
    w_res = w_dst / 360  # pixels per degree
    offset = int(adjusted_angle * w_res)

    x_src_ = np.roll(x_src_, shift=offset, axis=1)
    y_src_ = np.roll(y_src_, shift=offset, axis=1)
    # remap images using cv2
    map_x = x_src_.astype(np.float32)
    map_y = y_src_.astype(np.float32)
    return map_x, map_y


def upchannel(input):

    h, w = input.shape[:2]
    output = np.zeros([h, w, 3])
    for i in range(h):
        for j in range(w):
            d = input[i, j]
            output[i, j, 0] = d
    return output


def downchannel(input):
    h, w = input.shape[:2]
    output = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            d = input[i, j, 0]
            output[i, j] = d
    return output


def dualfisheye2cube(frame, aperture=203, face_w=256):
    aperture = aperture * np.pi / 180
    h_src, w_src = frame.shape[:2]
    w_dst, h_dst = [2000, 1000]

    # 创建网格坐标
    x_dst_norm_L, y_dst_norm_L = np.meshgrid(np.linspace(1, -1, w_dst),
                                             np.linspace(-1, 1, h_dst))
    x_dst_norm_R, y_dst_norm_R = np.meshgrid(np.linspace(-1, 1, w_dst),
                                             np.linspace(-1, 1, h_dst))

    r_L, theta_L = equicoortofisheyecoor(x_dst_norm_L, y_dst_norm_L, aperture)
    r_R, theta_R = equicoortofisheyecoor(x_dst_norm_R, y_dst_norm_R, aperture)
    x_src_norm_L = r_L * np.cos(theta_L)+1
    x_src_norm_R = r_R * np.cos(theta_R)-1
    y_src_norm_L = r_L * np.sin(theta_L)
    y_src_norm_R = r_R * np.sin(theta_R)

    x_src_L = lerp(0, w_src/2, 0, 2, x_src_norm_L)
    x_src_R = lerp(w_src/2, w_src, 0, -2, x_src_norm_R)
    y_src_L = lerp(0, h_src, -1, 1, y_src_norm_L)
    y_src_R = lerp(0, h_src, -1, 1, y_src_norm_R)

    # supppres out of the bound index error (warning this will overwrite multiply pixels!)
    x_src_L_ = np.minimum(w_src - 1, np.floor(x_src_L).astype(int))
    x_src_R_ = np.minimum(w_src - 1, np.floor(x_src_R).astype(int))
    x_src_ = np.concatenate(
        (x_src_L_[:, :int(w_dst/2)], x_src_R_[:, int(w_dst/2):]), axis=1)

    y_src_L_ = np.minimum(h_src - 1, np.floor(y_src_L).astype(int))
    y_src_R_ = np.minimum(h_src - 1, np.floor(y_src_R).astype(int))
    y_src_ = np.concatenate(
        (y_src_L_[:, :int(w_dst/2)], y_src_R_[:, int(w_dst/2):]), axis=1)

    # from equi to cube map

    x = upchannel(x_src_)
    y = upchannel(y_src_)

    cube_x = e2c(x, face_w=face_w)
    cube_y = e2c(y, face_w=face_w)

    x_src = downchannel(cube_x)
    y_src = downchannel(cube_y)

    x_src_ = np.minimum(w_src - 1, np.floor(x_src).astype(int))
    y_src_ = np.minimum(h_src - 1, np.floor(y_src).astype(int))
    # 使用cv2.remap函数重构图像
    map_x = x_src_.astype(np.float32)
    map_y = y_src_.astype(np.float32)
    return map_x, map_y


def equi2cube(frame, face_w=256):
    h_src, w_src = frame.shape[:2]
    x_src, y_src = np.meshgrid(np.linspace(0, w_src, w_src),
                               np.linspace(0, h_src, h_src))
    x = upchannel(x_src)
    y = upchannel(y_src)
    cube_x = e2c(x, face_w=face_w)
    cube_y = e2c(y, face_w=face_w)
    x_dst = downchannel(cube_x)
    y_dst = downchannel(cube_y)
    map_x = x_dst.astype(np.float32)
    map_y = y_dst.astype(np.float32)
    return map_x, map_y
