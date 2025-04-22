import requests


class OSM_Query:
    def __init__(self,
                 bbox=(54.7607, -1.5930, 54.7835, -1.5591),
                 feature_type='highway'):
        feature_types = ['highway', 'street_side_parking']
        assert (feature_type in feature_types)
        '''
        To do: add more types in the overpass_query than highway
        '''

        self.bbox = bbox
        overpass_url = "http://overpass-api.de/api/interpreter"
        if feature_type == 'highway':
            overpass_query = f"""
            [out:json];
            (way[highway~"^(service|primary|secondary|tertiary|residential|unclassified|trunk|trunk_link)$"]{self.bbox};
            );
            out geom;
            """
        elif feature_type == 'street_side_parking':
            overpass_query = f"""
            [out:json];
            (way[parking="street_side"]{bbox};
            );
            out geom;
            """
        response = requests.get(overpass_url,
                                params={'data': overpass_query})
        self.data = response.json()

    def get_elements(self):

        return self.data['elements']

    # get longitudes and latitudes from geojson and save to lists
    def get_range(self):
        longitude_list = []
        latitude_list = []
        elements = self.data['elements']
        for element in elements:
            #properties = feature['tags']
            #ind = properties['@id'][4:]
            geometry = element['geometry']

            for coor in geometry:
                latitude = coor['lat']
                longitude = coor['lon']
                longitude_list.append(longitude)
                latitude_list.append(latitude)

        if len(longitude_list) == len(latitude_list):
            min_long, max_long = min(longitude_list), max(longitude_list)
            min_lat, max_lat = min(latitude_list), max(latitude_list)
            print('Loaded elements:', len(elements))
            print('Min longitude:', min_long)
            print('Max longitude:', max_long)
            print('Min latitude:', min_lat)
            print('Max latitude:', max_lat)
            print('Longitude range:', max_long - min_long)
            print('Latitude range:', max_lat - min_lat)
            map_range = [min_lat, max_lat,
                         min_long, max_long]  # min_lat, max_lat, min_long, max_long

            return map_range
