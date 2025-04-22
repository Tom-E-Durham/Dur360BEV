"""
Module: Show Map
Author: Wenke E (wenke.e@durham.ac.uk)
Version: 1.0
"""

import json
import numpy as np
import math
import cv2
import os
import requests

# deal with the json code downloaded from the overpass turbo


def get_features(bbox=(54.7607, -1.5930, 54.7835, -1.5591)):
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    (way[highway~"^(service|primary|secondary|tertiary|residential|unclassified|trunk|trunk_link)$"]{bbox};
    );
    out geom;
    """
    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    data = response.json()
    return data['elements']

# get longitudes and latitudes from geojson and save to lists


def get_LL(features):
    longitude_list = []
    latitude_list = []

    for feature in features:
        properties = feature['tags']
        #ind = properties['@id'][4:]

        geometry = feature['geometry']

        for coor in geometry:
            latitude = coor['lat']
            longitude = coor['lon']
            longitude_list.append(longitude)
            latitude_list.append(latitude)

    if len(longitude_list) == len(latitude_list):
        min_long, max_long = min(longitude_list), max(longitude_list)
        min_lat, max_lat = min(latitude_list), max(latitude_list)
        print('total number of features:', len(features))
        print('min longitude:', min_long)
        print('max longitude:', max_long)
        print('min latitude:', min_lat)
        print('max latitude:', max_lat)
        print('longitude range:', max_long - min_long)
        print('latitude range:', max_lat - min_lat)
        map_range = [min_lat, max_lat,
                     min_long, max_long]  # min_long, max_long, min_lat, max_lat
        return longitude_list, latitude_list, map_range


def rad(degree):
    return degree / 180 * np.pi


def get_map(features, map_range):
    """
    Get the global map image using the geojson file
    """
    # set the background image size
    # Map range
    min_lat, max_lat, min_lon, max_lon = map_range
    # Constants
    EARTH_RADIUS = 6371000  # meters

    scale = 10  # 1 meter = 10 pixels in the global map
    # Calculate distances
    delta_lat = rad(max_lat - min_lat)
    delta_lon = rad(max_lon - min_lon)
    lat_distance = delta_lat * EARTH_RADIUS
    lon_distance = delta_lon * EARTH_RADIUS * np.cos(rad(min_lat+max_lat) / 2)
    map_res = [int(lat_distance * scale),
               int(lon_distance * scale),
               3]
    print('generating map with size :', map_res)
    global_map = np.zeros(map_res)

    # white color in RGB
    color = (255, 255, 255)
    thickness = int(3.65 * scale)  # UK lane 3.65m = 36.5 pixels
    for feature in features:
        properties = feature['tags']
        try:
            lanes = int(properties['lanes'])
        except:
            lanes = 1

        coordinates = np.array(feature['geometry'])
        latitudes = [coor['lat'] for coor in coordinates]
        longitudes = [coor['lon'] for coor in coordinates]

        dist_x, dist_y = LL2XY(latitudes, longitudes,
                               map_range, map_res=map_res)

        if len(dist_x) == len(dist_y) > 1:
            for i in range(len(dist_x)-1):
                point_a = [dist_x[i], dist_y[i]]
                point_b = [dist_x[i+1], dist_y[i+1]]
                global_map = cv2.line(
                    global_map, point_a, point_b, color, thickness*lanes)

                # draw the lane signs
                if lanes > 1:
                    global_map = cv2.line(
                        global_map, point_a, point_b, (0, 0, 255), 5)
    return global_map, map_res


def LL2XY(latitude, longitude, bbox, map_res):
    """
    Change the Lat, Lon coordinates to the X, Y coordinates on the map
    """
    # Aspect ratio correction based on spherical Earth approximation
    lat_res, lon_res = map_res[:2]

    min_lat, max_lat, min_lon, max_lon = bbox

    if type(longitude) == type(latitude) == list or type(longitude) == type(latitude) == np.ndarray:
        longitude = np.array(longitude)
        latitude = np.array(latitude)
        # Calculate the relative position of the longitude and latitude
        lon_ratio = (longitude - min_lon) / (max_lon - min_lon)
        lat_ratio = (latitude - min_lat) / (max_lat - min_lat)
        # Convert the relative position to pixel coordinates
        x = (lon_ratio * lon_res).astype(int)
        # For y, subtract from height because image coordinates start from the top left corner
        y = (lat_res - (lat_ratio * lat_res)).astype(int)

    # for the single point from dataset
    else:
        lon_ratio = (longitude - min_lon) / (max_lon - min_lon)
        lat_ratio = (latitude - min_lat) / (max_lat - min_lat)
        x = (lon_ratio * lon_res).astype(int)
        y = (lat_res - (lat_ratio * lat_res)).astype(int)

    return x, y


def get_localmap(global_map, location, res=[200, 200]):
    map_range = global_map.shape[0]
    #size = int(map_range / zoom)
    x, y = location

    res_w = int(res[0]*np.sqrt(2)*5)
    res = [res_w, res_w]
    left = min(int(x-res[0]/2), map_range-res[0])
    bottom = min(int(y-res[1]/2), map_range - res[1])
    # get 1414x1414 pixels, 1 meter = 10 pixels
    local_map = global_map[bottom: bottom + res[1], left:left+res[0]]
    # 282 = 200*sqrt(2) for rotation
    local_map = cv2.resize(local_map, [282, 282], interpolation=cv2.INTER_AREA)

    return local_map


def draw_car(local_map, res=[200, 200]):
    # twizy size (in meters):
    ego_L = 2.23
    ego_W = 1.19
    ego_H = 1.46

    # local map range: 200x200 Pixels, 100x100 meters, 1 meter = 2 pixels
    h, w = ego_L*2, ego_W*2
    center = [int(local_map.shape[0]/2), int(local_map.shape[0]/2)]
    top_left = (center[0]-int(w/2), center[1]-int(h/2))
    bottom_right = (center[0]+int(w/2), center[1]+int(h/2))
    local_map_plotted = cv2.rectangle(
        local_map, top_left, bottom_right, (128, 0, 128), cv2.FILLED)
    return local_map_plotted


def initialization():
    features = get_features()
    _, _, map_range = get_LL(features)

    global_map, map_res = get_map(features, map_range)
    cv2.imwrite('./global_map.png', global_map)
    return global_map, map_range, map_res


def show_map(LL, global_map, map_range, map_res, yaw, res=[200, 200]):
    """
    Return the final local map rotated and ego car plotted
    """
    lat, lon = LL
    # get local map using lat, lon
    x, y = LL2XY(lat, lon, map_range, map_res)

    local_map = get_localmap(global_map, [x, y], res)
    alpha = 2 / np.sqrt(2)
    res_ = [int(res[0]*alpha), int(res[1]*alpha)]

    yaw = 90 - yaw * 180 / np.pi
    M = cv2.getRotationMatrix2D(
        (int(res_[0]/2), int(res_[1]/2)), int(yaw), 1.0)

    start_w = (res_[0]-res[0])//2
    rotated = cv2.warpAffine(local_map, M, res_)[
        start_w: start_w+res[0], start_w: start_w+res[1], :].astype(np.uint8)
    local_map_plotted = draw_car(rotated, res)
    return local_map_plotted
