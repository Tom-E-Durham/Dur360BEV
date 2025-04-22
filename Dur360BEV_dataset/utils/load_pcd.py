import numpy as np
import struct


class PointCloud_Loader:
    def __init__(self):
        pass

    def kitti(self, file_path):
        point_cloud_data = np.fromfile(
            file_path, '<f4')  # little-endian float32
        points = np.reshape(point_cloud_data, (-1, 4))   # x, y, z, r
        return points  # [n , 4]

    def pcd(self, file_path, all_field=False, if_reflectivity=True):
        header = {}
        data_start = 0  # set a val to count the length of header
        with open(file_path, 'rb') as f:
            while True:
                line = f.readline()
                data_start += len(line)
                if line.startswith(b'#') or len(line) == 0:
                    continue
                if line.startswith(b'DATA'):
                    break
                line = line.strip().decode('utf-8')
                key, value = line.split(' ', 1)
                header[key] = value
            point_size = sum(int(size) * int(count) for size,
                             count in zip(header['SIZE'].split(), header['COUNT'].split()))
            num_points = int(header['POINTS'])
            # Read the rest of the file (binary data)
            binary_data = f.read(point_size * num_points)

        points = []  # List to hold the point data (x, y, z, intensity)
        # Process the binary data
        for i in range(int(header['POINTS'])):
            offset = i * point_size  # Calculate the offset for the current point
            point_data = binary_data[offset:offset+point_size]
            x, y, z = struct.unpack_from('fff', point_data, 0)
            intensity = struct.unpack_from('f', point_data, 16)
            reflectivity = struct.unpack_from('H', point_data, 24)
            if all_field:
                if_reflectivity = False
                t = struct.unpack_from('I', point_data, 20)
                ring = struct.unpack_from('B', point_data, 26)
                ambient = struct.unpack_from('H', point_data, 28)
                range_ = struct.unpack_from('I', point_data, 32)
                # intensity is a tuple, get the first element
                points.append(
                    (x, y, z, intensity[0], t[0], reflectivity[0], ring[0], ambient[0], range_[0]))
            elif if_reflectivity:
                points.append((x, y, z, reflectivity[0]))
            else:
                points.append((x, y, z, intensity[0]))
        points = np.asarray(points)
        return points

    def nuscenes(self, file_path, all_field=False):
        scan = np.fromfile(file_path, dtype=np.float32)
        if all_field:
            points = scan.reshape((-1, 5))
        else:
            points = scan.reshape((-1, 5))[:, :4]
        return points
