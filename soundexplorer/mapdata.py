import pathlib

import geopandas
import numpy as np
import pyproj
import requests
import shapely
import tqdm


class MapData:
    """
    Class to het the spatial data. Default is habitat type
    """

    def __init__(self, map_path=None):
        """

        Parameters
        ----------
        map_path : string or Path
            Path where the data is
        """
        if map_path is None:
            map_path = pathlib.Path('//fs/shared/datac/Geo/Layers/Belgium/habitatsandbiotopes/'
                                    'broadscalehabitatmap/seabedhabitat_BE.shp')
        else:
            if not isinstance(map_path, pathlib.Path):
                map_path = pathlib.Path(map_path)
        self.map_path = map_path
        self.map = geopandas.read_file(self.map_path)

    def get_location_map_data(self, columns, location, crs):
        """
        Get the features of the columns at a certain location
        Parameters
        ----------
        columns : list of strings
            Columns to get the features from
        location : geometry object
            Location
        crs : string or object
            CRS projection

        Returns
        -------
        List of features in that particular location
        """
        project = pyproj.Transformer.from_crs(crs.geodetic_crs, self.map.crs, always_xy=True).transform
        location_crs = shapely.ops.transform(project, location)
        mask = self.map.contains(location_crs)
        try:
            idx = np.where(mask)[0][0]
            features = self.map.loc[idx, columns].values
        except IndexError:
            print('This location is not included in the map, setting to Nan')
            features = np.nan
        return features

    def get_seabottom_data(self, df, columns):
        """
        Get the features and the bathymetry together of all the DataFrame
        Parameters
        ----------
        df : DataFrame
            Where to add the seabottom data
        columns : list of strings
            Columns from the map to add
        Returns
        -------
        DataFrame with features and bathymetry added
        """
        for column in columns:
            df[(column, 'all')] = None
        diff_points = df.geometry.unique()
        for point in tqdm(diff_points):
            if point is not None:
                if len(diff_points) == 1:
                    idxes = df.index
                else:
                    idxes = df[df[('geometry', '')] == point].index
                df.loc[idxes, (columns, 'all')] = self.get_location_map_data(columns, point, crs=df.crs)
                df.loc[idxes, ('bathymetry', 'all')] = self.get_bathymetry(point)
            else:
                idxes = df[df[('geometry', '')] == None].index
                df.loc[idxes, (columns, 'all')] = np.nan
                df.loc[idxes, ('bathymetry', 'all')] = np.nan
        if len(diff_points) == 0:
            df[('bathymetry', 'all')] = np.nan
        return df

    @staticmethod
    def get_bathymetry(location):
        """
        Return the bathymetry of a certain point from EMODnet
        Parameters
        ----------
        location : geometry object
            Location

        Returns
        -------
        Bathymetry as a positive number
        """
        point_str = str(location)
        response = requests.get("https://rest.emodnet-bathymetry.eu/depth_sample?geom=%s" % point_str)
        if response.status_code == 200:
            depth = response.json()['avg']
        else:
            depth = 0.0
        return depth
