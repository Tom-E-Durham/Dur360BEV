from . import getcvmap
import cv2


def df2c(input, aperture=203, face_w=256):
    map_x, map_y = getcvmap.dualfisheye2cube(
        input, aperture=aperture, face_w=face_w)
    dst_img = cv2.remap(input, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return dst_img
