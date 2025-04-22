from . import getcvmap
import cv2


def df2e(input, size=[2000, 1000], aperture=203, center_angle=0):
    map_x, map_y = getcvmap.dualfisheye2equi(
        input, size=size, aperture=aperture, center_angle=center_angle)
    dst_img = cv2.remap(input, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return dst_img
