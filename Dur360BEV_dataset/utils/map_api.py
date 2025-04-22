import numpy as np
import pandas as pd
import requests
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString, box
from shapely.ops import unary_union, nearest_points
from shapely.plotting import plot_polygon, plot_line
from .query import OSM_Query


class OSMSemanticMap:
    """
    OSMSemanticMap database class for dealing with the data queried from OpenStreetMap and generate the semantic map.
    """

    def __init__(self,
                 bbox=(54.7607, -1.5930, 54.7835, -1.5591)):

        # load the data within the bounding box
        query = OSM_Query(bbox)
        self.bbox = bbox
        self.elements = query.get_elements()
        self.map_range = query.get_range()  # min_lat, max_lat, min_long, max_long
        self.geodf = self.create_geodf()

    # Convert road data to GeoDataFrame
    def create_geodf(self):
        geometries = []
        ids = []
        names = []
        lanes = []
        sidewalks = []
        for road in self.elements:
            line_points = [Point(xy['lon'], xy['lat'])
                           for xy in road['geometry']]
            geometries.append(LineString(line_points))
            ids.append(road['id'])

            # get road names
            try:
                road_name = road['tags']['name']
            except:
                road_name = None
            names.append(road_name)

            # get lane numbers
            try:
                lane_num = int(road['tags']['lanes'])
            except:
                lane_num = 1
            lanes.append(lane_num)

            # get sidewalk
            try:
                sidewalk = road['tags']['sidewalk']
            except:
                sidewalk = 'no'
            sidewalks.append(sidewalk)
            #
        geodf = gpd.GeoDataFrame({'road_id': ids, 'geometry': geometries, 'road_name': names,
                                  'lanes': lanes, 'sidewalk': sidewalks}, geometry='geometry')
        geodf.set_index('road_id', inplace=True)
        return geodf

    def get_closest_road(self, LL):
        lon, lat = LL
        # Define GPS point of interest
        self.current_point = LL
        gps_point = Point(lon, lat)  # Example coordinates near the road

        # Calculate the distance to each road
        self.geodf['distance'] = self.geodf.apply(
            lambda row: row['geometry'].distance(gps_point), axis=1)

        # Find the road with the minimum distance to the GPS point
        closest_road_id = self.geodf['distance'].idxmin()
        return closest_road_id

    def get_road(self, road_id_to_find):
        return self.geodf.loc[road_id_to_find]


class OSMSemanticMapVis:
    """
    Visualisation tool to display the OSMSemanticMap data
    """

    def __init__(self,
                 map_api: OSMSemanticMap,
                 scale=1):

        self.map_api = map_api
        self.scale = scale  # scale: pixel number for 1 meter in the global map
        self.lane_width = 3.65  # single lane = 3.65 meters in the UK
        self.generate_map_info()

    def rad(self, degree):
        return degree / 180 * np.pi

    def perpendicular_vector(sefl, v):
        """ Return a vector that is perpendicular to the given 2D vector `v`. """
        perp = np.array([-v[1], v[0]])
        EPS = 1e-6
        return perp / np.linalg.norm(perp + EPS)

    def LL2XY(self, longitude, latitude):
        """
        Change the Lat, Lon coordinates to the X, Y coordinates on the map
        Measured based on the map resolution and the map LL range.
        """
        # Aspect ratio correction based on spherical Earth approximation
        lat_res, lon_res = self.map_res[:2]

        min_lat, max_lat, min_lon, max_lon = self.map_api.map_range

        if type(longitude) == type(latitude) == list or type(longitude) == type(latitude) == np.ndarray:
            longitude = np.array(longitude)
            latitude = np.array(latitude)
            # Calculate the relative position of the longitude and latitude
            lon_ratio = (longitude - min_lon) / (max_lon - min_lon)
            lat_ratio = (latitude - min_lat) / (max_lat - min_lat)
            # Convert the relative position to pixel coordinates
            #x = (lon_ratio * lon_res).astype(int)
            x = lon_ratio * lon_res
            # For y, subtract from height because image coordinates start from the top left corner
            #y = (lat_res - (lat_ratio * lat_res)).astype(int)
            y = lat_ratio * lat_res
        # for the single point from dataset

        else:
            lon_ratio = (longitude - min_lon) / (max_lon - min_lon)
            lat_ratio = (latitude - min_lat) / (max_lat - min_lat)
            x = lon_ratio * lon_res
            #y = (lat_res - (lat_ratio * lat_res)).astype(int)
            y = lat_ratio * lat_res
        return x, y

    def generate_map_info(self):
        min_lat, max_lat, min_lon, max_lon = self.map_api.map_range
        # Constants
        EARTH_RADIUS = 6371000  # meters

        scale = 1  # 1 meter = 1 pixel in the global map
        # Calculate distances using haversine formula
        delta_lat = self.rad(max_lat - min_lat)
        delta_lon = self.rad(max_lon - min_lon)
        lat_distance = delta_lat * EARTH_RADIUS
        lon_distance = delta_lon * EARTH_RADIUS * \
            np.cos(self.rad(min_lat+min_lat) / 2)

        map_res = [int(lat_distance * scale),
                   int(lon_distance * scale),
                   3]
        self.map_res = map_res

    def get_center_lines(self):
        road_x_list = []
        road_y_list = []
        for feature in self.map_api.elements:
            coordinates = np.array(feature['geometry'])
            latitudes = [coor['lat'] for coor in coordinates]
            longitudes = [coor['lon'] for coor in coordinates]

            road_x, road_y = self.LL2XY(longitudes, latitudes)
            road_x_list.append(road_x)
            road_y_list.append(road_y)

        return road_x_list, road_y_list

    def get_lane_nums(self):
        return np.array(self.map_api.geodf['lanes'])

    def get_center_lines_LL(self):
        lines = list(self.map_api.geodf['geometry'])
        lines = unary_union(lines)
        return lines

    def get_orientation(self, v, p):
        det = v[0] * p[1] - v[1] * p[0]
        orientation = "left" if det > 0 else "right"
        return orientation

    def fit_point(self, lon, lat, yaw):
        LL = [lon, lat]
        closest_road_id = self.map_api.get_closest_road(LL)
        lane_lines = self.map_api.get_road(closest_road_id)['lane_geometry']
        center_line_LL = self.map_api.get_road(closest_road_id)['geometry']
        center_line_xy = self.LL2XY(np.asarray(
            center_line_LL.xy[0]), np.asarray(center_line_LL.xy[1]))
        center_line = LineString(np.array(center_line_xy).T)
        car_x, car_y = self.LL2XY(LL[0], LL[1])
        car_point = Point(car_x, car_y)
        car_on_road = nearest_points(car_point, center_line)[1]
        road_x = car_on_road.xy[0][0]
        road_y = car_on_road.xy[1][0]
        # get yaw vector from yaw info
        yaw_v = [np.cos(yaw), np.sin(yaw)]
        if len(lane_lines.geoms) > 1:
            left_lanes = []
            for lane in lane_lines.geoms:
                # lane: LineString obj
                closest_on_lane = nearest_points(car_on_road, lane)[1]
                point_x = closest_on_lane.xy[0][0]
                point_y = closest_on_lane.xy[1][0]
                dir_v = [point_x-road_x, point_y-road_y]
                if self.get_orientation(yaw_v, dir_v) == 'left':
                    left_lanes.append(lane)
            if len(left_lanes) > 1:
                left_lanes = MultiLineString(left_lanes)
            else:
                left_lanes = left_lanes[0]
            fitted_point = nearest_points(car_on_road, left_lanes)[1]
        else:
            fitted_point = nearest_points(car_on_road, lane_lines)[1]

        return fitted_point

    def get_street_side_parking(self):
        query_parking = OSM_Query(
            self.map_api.bbox, feature_type='street_side_parking')
        elements = query_parking.get_elements()
        parking_x_list = []
        parking_y_list = []
        for element in elements:
            coordinates = np.array(element['geometry'])
            latitudes = [coor['lat'] for coor in coordinates]
            longitudes = [coor['lon'] for coor in coordinates]

            parking_x, parking_y = self.LL2XY(longitudes, latitudes)
            parking_x_list.append(parking_x)
            parking_y_list.append(parking_y)
        parking_areas = []
        for idx in range(len(parking_x_list)):
            parking_points = np.stack(
                (parking_x_list[idx], parking_y_list[idx]), axis=1)
            parking_area = Polygon(parking_points)
            parking_areas.append(parking_area)
        parking_areas = MultiPolygon(parking_areas)
        return parking_areas

    def render_map(self, layers=['driving_area', 'road_centerline', 'lane_divider']):
        road_x_list, road_y_list = self.get_center_lines()
        lane_numbers = self.get_lane_nums()

        side_walks = np.array(self.map_api.geodf['sidewalk'])
        offset_dict = {'both': [-1, 1], 'left': [1], 'right': [-1]}

        road_polygons = []
        c_lines = []
        lane_dividers = []
        lane_centers = []
        side_walk_centers = []
        lane_width = 3.65
        side_walk_width = 1.5
        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        for i in range(len(road_x_list)):
            x_arr = road_x_list[i]
            y_arr = road_y_list[i]
            points = np.stack((x_arr, y_arr), axis=1)
            cur_road = []
            cur_lane_dividers = []
            cur_lane_centers = []
            num_lanes = lane_numbers[i]

            if 'side_walk' in layers:
                side_walk = side_walks[i]
                if side_walk in offset_dict.keys():
                    offsets = offset_dict[side_walk]
                    for offset in offsets:
                        cur_side_walk = []
                        for index in range(len(points)-1):
                            cur_points = points[index:index+2]
                            direction = np.diff(cur_points, axis=0)
                            perp = np.array(
                                self.perpendicular_vector(direction[0]))
                            sidewalk_points = cur_points + perp * \
                                (lane_width * num_lanes/2 +
                                 side_walk_width) * offset
                            cur_side_walk.extend(
                                [list(sidewalk_points[0]), list(sidewalk_points[1])])
                        # .buffer(1, cap_style=2).buffer(-1, cap_style=2)
                        cur_side_walk = LineString(cur_side_walk)
                        side_walk_centers.append(cur_side_walk)

            for i in range(1, num_lanes + 1):
                cur_lane_center = []
                for index in range(len(points)-1):
                    cur_points = points[index:index+2]
                    direction = np.diff(cur_points, axis=0)
                    perp = np.array(self.perpendicular_vector(direction[0]))

                    # Calculate shifted points for the left and right side of each lane
                    left_offsets = cur_points + perp * \
                        (lane_width * (num_lanes/2 - i)+lane_width)
                    right_offsets = cur_points + perp * \
                        (lane_width * (num_lanes/2 - i))

                    if i != num_lanes:
                        lane_divider = right_offsets
                        lane_divider = LineString(lane_divider)

                    # Combine to form a closed polygon: traverse down one side and back the other
                    polygon = Polygon(
                        np.vstack([left_offsets, right_offsets[::-1]]))
                    cur_road.append(polygon)
                    cur_lane_dividers.append(lane_divider)

                    lane_center = cur_points + perp * \
                        (lane_width * (num_lanes/2 - i)+lane_width/2)

                    cur_lane_center.extend(
                        [list(lane_center[0]), list(lane_center[1])])

                #uni_cur_lane_center = [cur_lane_center[0]] + [cur_lane_center[i] for i in range(1, len(cur_lane_center)) if cur_lane_center[i] != cur_lane_center[i - 1]]
                # print(len(uni_cur_lane_center))

                cur_lane_center = LineString(cur_lane_center)
                cur_lane_centers.append(cur_lane_center)

            # Combine the current polygons and lines
            cur_road = MultiPolygon(cur_road).buffer(
                5, cap_style='flat').buffer(-5, cap_style='flat')
            road_polygons.append(cur_road)

            cur_lane_dividers = unary_union(cur_lane_dividers)
            lane_dividers.append(cur_lane_dividers)

            cur_lane_centers = MultiLineString(cur_lane_centers)
            lane_centers.append(cur_lane_centers)

            c_lines.append(LineString(points))

        polygons = MultiPolygon(road_polygons)
        polygons = unary_union(polygons)
        polygons = polygons.buffer(
            lane_width/2, cap_style='flat').buffer(-lane_width/2, cap_style='flat')

        lane_dividers = unary_union(lane_dividers)

        self.map_api.geodf['lane_geometry'] = lane_centers
        lane_centers = unary_union(lane_centers)
        c_lines = unary_union(c_lines)

        side_walk_centers = MultiLineString(side_walk_centers)
        side_walk_centers = unary_union(side_walk_centers)
        side_walk_centers = side_walk_centers.buffer(
            side_walk_width*2).buffer(-side_walk_width)
        # render
        if 'side_walk' in layers:
            plot_polygon(side_walk_centers, ax=ax, add_points=False,
                         color='red', alpha=0.5, label='Side walk path')
        if 'driving_area' in layers:
            plot_polygon(polygons, ax=ax, add_points=False,
                         alpha=0.7, label='Driving Area')
        if 'road_centerline' in layers:
            plot_line(c_lines, ax=ax, add_points=False,
                      linewidth=1, label='Road Center Line')
        if 'lane_divider' in layers:
            plot_line(lane_dividers, ax=ax, add_points=False, linewidth=1,
                      color='white', linestyle='--', label='Lane divider Line')
        if 'street_side_parking' in layers:
            parking_polygons = self.get_street_side_parking()
            plot_polygon(parking_polygons, ax=ax, add_points=False,
                         color='yellow', alpha=0.5, label='Street side parking area')
        #plot_line(lane_centers, ax=ax, add_points=False, linewidth=1, linestyle = '-', color = 'green', label='Lane center Line')
        # Set plot attributes
        ax.set_aspect('equal')
        ax.legend()

        self.polygons = polygons
        return fig, ax

    def _render_local_map(self, gps_path, odom_path, layers=['driving_area', 'road_centerline', 'lane_divider'], if_fit=True):
        # prepare the global map
        fig, ax = self.render_map(layers)
        # Disable autoscaling
        ax.set_autoscale_on(False)

        # load gps data
        gps_path = './revised_gps_data.json'
        gps_df = pd.read_json(gps_path)
        lat, lon = np.array(gps_df.iloc[:, :2]).T

        # load odom data to get yaw info
        yaw_arr = np.array(pd.read_json(odom_path).iloc[:, 2])

        if if_fit:
            # fit gps on lane center
            fit_point = self.fit_point(lon[0], lat[0], yaw_arr[0])
            gps_x, gps_y = fit_point.xy
        else:
            gps_x, gps_y = self.LL2XY(lon[0], lat[0])

        # Initialize the scatter plot with the first point
        car_scat = ax.scatter(gps_x, gps_y, s=50,
                              color='red', marker='x', label='Ego Car')

        def update(frame_number):
            print(f"[DEBUG INFO] Fitting point on lane center: {frame_number}")
            if if_fit:
                fit_point = self.fit_point(
                    lon[frame_number], lat[frame_number], yaw_arr[frame_number])
                gps_x = fit_point.xy[0][0]
                gps_y = fit_point.xy[1][0]
            else:
                gps_x, gps_y = self.LL2XY(lon[frame_number], lat[frame_number])
            # Update the data for the scatter plot
            car_scat.set_offsets([gps_x, gps_y])

            # Make sure the point remains visible
            car_scat.set_visible(True)

            # Calculate new axis limits
            new_x_limits = (gps_x - 50, gps_x + 50)
            new_y_limits = (gps_y - 50, gps_y + 50)

            # Update the axis limits to follow the car
            ax.set_xlim(new_x_limits)  # 100m range centered on the car
            ax.set_ylim(new_y_limits)  # 100m range centered on the car

            return car_scat,
        ani = FuncAnimation(fig, update, frames=len(lon),
                            blit=False, repeat=True)

        return fig, ax, ani

    def get_local_bin(self, gps_location, yaw=None, search_range=100, map_scale=2):
        """
        given gps location, search local elements with given range in meters (search_range, search_range)
        output binary map with the resolution: search_range * map_scale
        """
        # prepare global map's polygons
        if not hasattr(self, 'polygons'):
            _, _ = self.render_map(['driving_area'])
        global_polygons = self.polygons

        lon, lat = gps_location
        center_x, center_y = self.LL2XY(lon, lat)

        # final output resolution: search_range(m)*map_scale(pix/m)
        dist_res = search_range * map_scale
        # in map space, for maximum rotation: 45 degree
        clip_range = search_range * self.scale * np.sqrt(2)

        # get clip range for search window in map space
        start_x = center_x - clip_range // 2
        start_y = center_y - clip_range // 2
        end_x = center_x + clip_range // 2
        end_y = center_y + clip_range // 2

        # clip the global polygons with given gps location and clip range
        search_area = box(start_x, start_y, end_x, end_y)
        local_polygons = global_polygons.intersection(search_area)

        # draw polygons on bin_img with cv2
        clip_res = int(clip_range*map_scale)
        binary_img = np.zeros((clip_res, clip_res), dtype=np.uint8)

        if not local_polygons.is_empty:
            if isinstance(local_polygons, Polygon):
                exterior_coords = np.array([((x-start_x)*map_scale, clip_res-(y-start_y)*map_scale)
                                            for x, y in local_polygons.exterior.coords], dtype=np.int32)
                cv2.fillPoly(binary_img, [exterior_coords], 1)
                for interior in local_polygons.interiors:
                    interior_coords = np.array([((x-start_x)*map_scale, clip_res-(
                        y-start_y)*map_scale) for x, y in interior.coords], dtype=np.int32)
                    cv2.fillPoly(binary_img, [interior_coords], 0)
            elif isinstance(local_polygons, MultiPolygon):
                for polygon in local_polygons.geoms:
                    exterior_coords = np.array([((x-start_x)*map_scale, clip_res-(
                        y-start_y)*map_scale) for x, y in polygon.exterior.coords], dtype=np.int32)
                    cv2.fillPoly(binary_img, [exterior_coords], 1)
                    for interior in polygon.interiors:
                        interior_coords = np.array([((x-start_x)*map_scale, clip_res-(
                            y-start_y)*map_scale) for x, y in interior.coords], dtype=np.int32)
                        cv2.fillPoly(binary_img, [interior_coords], 0)

        if yaw is not None:
            # rotate the binary image according to yaw
            yaw = 90 - yaw * 180 / np.pi
            M = cv2.getRotationMatrix2D((clip_res//2, clip_res//2), yaw, 1.0)
            rotated_img = cv2.warpAffine(
                binary_img, M, (clip_res, clip_res), flags=cv2.INTER_CUBIC)

        else:
            rotated_img = binary_img
        # clip the final output image
        output_img = rotated_img[clip_res//2 - dist_res//2: clip_res//2 + dist_res//2,
                                 clip_res//2 - dist_res//2: clip_res//2 + dist_res//2]
        return output_img


'''   
    def get_map(self):
        """
        Get the global map image using the geojson file
        """
        # set the background image size
        # Map range
        min_lat, max_lat, min_lon, max_lon = self.map_api.map_range
        # Constants
        EARTH_RADIUS = 6371000  # meters
        
        # Calculate distances using haversine formula
        delta_lat = self.rad(max_lat - min_lat)
        delta_lon = self.rad(max_lon - min_lon)
        lat_distance = delta_lat * EARTH_RADIUS
        lon_distance = delta_lon * EARTH_RADIUS * np.cos(self.rad(min_lat+min_lat) / 2)
        
        map_res = [int(lat_distance * self.scale), 
                int(lon_distance * self.scale),
                3]
        
        print('generating map with size :', map_res)
        global_map = np.zeros(map_res)

        # white color in RGB 
        color = (255, 255, 255)
        thickness = int(3.65 * self.scale)# UK lane 3.65m = 36.5 pixels
        
        for element in self.elements:
            properties = element['tags']
            try:
                lanes = int(properties['lanes'])
            except:
                lanes = 1

            coordinates = np.array(element['geometry'])
            latitudes = [coor['lat'] for coor in coordinates]
            longitudes = [coor['lon'] for coor in coordinates]
            
            dist_x, dist_y = self.LL2XY(latitudes, longitudes, map_res = map_res)

            if len(dist_x) == len(dist_y)>1:
                for i in range(len(dist_x)-1):
                    point_a = [dist_x[i], dist_y[i]]
                    point_b = [dist_x[i+1], dist_y[i+1]]
                    global_map = cv2.line(global_map, point_a, point_b, color, thickness*lanes)

                    # draw the lane signs
                    if lanes > 1:
                        global_map = cv2.line(global_map, point_a, point_b, (0,0,255), self.scale * 0.5)
        return global_map

    def get_localmap(self, global_map, location, res = [200,200]):
        map_range = global_map.shape[0]
        #size = int(map_range / zoom)
        x, y = location

        res_w = int(res[0]*np.sqrt(2)*5)
        res = [res_w,res_w] 
        left = min(int(x-res[0]/2), map_range-res[0])
        bottom = min(int(y-res[1]/2), map_range -res[1])
        local_map = global_map[bottom: bottom +res[1] ,left:left+res[0]] # get 1414x1414 pixels, 1 meter = 10 pixels
        local_map = cv2.resize(local_map, [282,282], interpolation=cv2.INTER_AREA) # 282 = 200*sqrt(2) for rotation
        
        return local_map
    
    def show_map(self, LL, global_map, yaw, res = [200, 200]):
        """
        Return the final local map rotated and ego car plotted
        """
        lat, lon = LL
        ## get local map using lat, lon
        map_res = global_map.shape[:2]
        x, y = self.LL2XY(lat, lon, self.map_range, map_res)

        local_map = self.get_localmap(global_map, [x,y], res)
        alpha = 2 / np.sqrt(2)
        res_ = [int(res[0]*alpha), int(res[1]*alpha)]
        
        yaw = 90 - yaw * 180 / np.pi 
        M = cv2.getRotationMatrix2D((int(res_[0]/2), int(res_[1]/2)), int(yaw), 1.0)
        
        start_w = (res_[0]-res[0])//2
        rotated = cv2.warpAffine(local_map, M, res_)[start_w: start_w+res[0], start_w: start_w+res[1],:].astype(np.uint8)
        local_map_plotted = self.draw_car(rotated, res)
        return local_map_plotted
'''
