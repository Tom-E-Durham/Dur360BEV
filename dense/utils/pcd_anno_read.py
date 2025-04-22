"""
Module: Point Cloud Annotation Reader
Author: Wenke E (wenke.e@durham.ac.uk)
Version: 1.0
"""

import json

anno_path = 'path/to/json'

with open(anno_path, 'r') as file:
    pcd_anno = json.load(file)

anno = pcd_anno[0]  # get from list
objects = anno['objects']

'''
Format for each object:

{'id': 'E9472640-78C5-4911-AD82-D967B7A46B2D',
 'type': '3D_BOX',
 'classId': None,
 'className': None,
 'trackId': 'Hj82CWRchtBrXWqy',
 'trackName': '1',
 'classValues': [],
 'contour': {'pointN': 0,
            'points': [],
            'size3D': {'x': 4.948, 'y': 2.07, 'z': 2.042},
            'center3D': {'x': 3.106, 'y': -2.704, 'z': -0.43},
            'viewIndex': 0,
            'rotation3D': {'x': 0, 'y': 0, 'z': -2.779}},
 'modelConfidence': 0.808,
 'modelClass': 'Car'}

'''


def get_objects(objects):
    labels = []
    center3Ds = []
    rots = []
    sizes = []
    for object in objects:
        label = object['modelClass']
        contour = object["contour"]

        center3D = contour['center3D']  # center point of the object
        rot = contour['rotation3D']['z']  # rotation on the BEV map

        # size of the object on x,y,z axes
        s_x, s_y, s_z = contour['size3D'].values()

        labels.append(label)
        center3Ds.append(center3D)
        rots.append(rot)
        sizes.append([s_x, s_y, s_z])
    return labels, center3Ds, rots, sizes


labels, center3Ds, rots, sizes = get_objects(objects)
print(len(labels), "objects in the annotation file.")
