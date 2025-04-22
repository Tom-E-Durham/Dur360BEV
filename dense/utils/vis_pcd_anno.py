"""
Module: Visualize Point Cloud Annotation
Author: Wenke E (wenke.e@durham.ac.uk)
Version: 1.0
"""

import math
import numpy as np
import cv2


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def draw_rotated_rectangle(img, center, width, height, angle, color):
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
        [rotate_point(center, pt, angle_rad) for pt in corners], np.int32)
    rotated_corners = rotated_corners.reshape((-1, 1, 2))

    # Draw the filled rotated rectangle
    cv2.fillPoly(img, [rotated_corners], color)


def add_objects_to_map(dist_map, anno, scale=2):
    '''
    scale: from meter to pixel. e.g. meters * scale = pixels in final map
    '''

    version = anno['version']
    if version == '1.0':
        objects = anno['instances']
    else:
        objects = anno['objects']
    for object in objects:
        if object['className']:
            label = object['className']  # from munual label
        else:
            label = object['modelClass']  # from model
        contour = object["contour"]

        center3D = contour['center3D']
        x, y, z = center3D.values()  # x : forward, y : left, z : up

        if (abs(x) < 2.5) and (abs(y) < 0.9) and (label == 'Car'):  # ignore the ergo car
            pass
        else:
            rot = contour['rotation3D']['z']

            w, h, _ = contour['size3D'].values()  # size of the object

            u = int(-y*scale+dist_map.shape[1]/2)
            v = int(-x*scale+dist_map.shape[1]/2)
            center = (u, v)
            # Replace with the rotation angle in degrees
            angle = math.degrees(-rot)+90
            if label == 'Pedestrian':
                draw_rotated_rectangle(
                    dist_map, center, w*scale, h*scale, angle, (255, 0, 0))
            elif label == 'Car' or label == 'Bus' or label == 'Truck':
                draw_rotated_rectangle(
                    dist_map, center, w*scale, h*scale, angle, (0, 255, 0))

    return dist_map
