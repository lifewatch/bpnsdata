import pathlib

import geopandas
import numpy as np
import pyproj
import requests
import shapely
from tqdm import tqdm


class MapData:
    """
    Class to het the spatial data. Default is habitat type
    """

    def __init__(self, map_path=None, borders_path=None, benthic_path=None):
        """

        Parameters
        ----------
        map_path: string or Path
            Path where the data is. If set to None the default is in VLIZ's archive (seabedhabitat)
        borders_path: string or Path
            Path where the borders of the desired zone to consider are. If set to None, default is Belgium EEZ.
        benthic_path: string or Path
            Path where the benthic habitat distribution is stored. If set to None, default is the ILVO map.
        """
        if map_path is None:
            map_path = pathlib.Path('//fs/shared/datac/Geo/Layers/Belgium/habitatsandbiotopes/'
                                    'broadscalehabitatmap/seabedhabitat_BE.shp')
        else:
            if not isinstance(map_path, pathlib.Path):
                map_path = pathlib.Path(map_path)

        if borders_path is None:
            borders_path = pathlib.Path('//fs/shared/datac/Geo/Layers/Belgium/administrativeunits/'
                                        'maritime_boundaries/MarineRegions/eez_boundaries_v10_BE_epsg4326.shp')
        else:
            if not isinstance(borders_path, pathlib.Path):
                borders_path = pathlib.Path(borders_path)

        if benthic_path is None:
            benthic_path = pathlib.Path('//fs/shared/onderzoek/6. Marine Observation Center/Projects/PhD_Clea/Data/maps'
                                        '/Habitat Suitability2/Habitat Suitability2.shp')
        else:
            if not isinstance(benthic_path, pathlib.Path):
                benthic_path = pathlib.Path(benthic_path)

        self.map_path = map_path
        self.map = geopandas.read_file(self.map_path)
        self.borders_path = borders_path
        self.borders = geopandas.read_file(self.borders_path)
        self.benthic_path = benthic_path
        self.benthic = geopandas.read_file(self.benthic_path)

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
        for point in tqdm(diff_points, total=len(diff_points)):
            if point is not None:
                if len(diff_points) == 1:
                    idxes = df.index
                else:
                    idxes = df[df[('geometry', '')] == point].index
                df.loc[idxes, (columns, 'all')] = self.get_location_map_data(self.map, columns, point, crs=df.crs)
                df.loc[idxes, ('benthic', 'all')] = self.get_location_map_data(self.benthic, 'GRIDCODE',
                                                                               point, crs=df.crs)
                df.loc[idxes, ('bathymetry', 'all')] = self.get_bathymetry(point)
            else:
                idxes = df[df[('geometry', '')].isnull()].index
                df.loc[idxes, (columns, 'all')] = np.nan
                df.loc[idxes, ('benthic', 'all')] = np.nan
                df.loc[idxes, ('bathymetry', 'all')] = np.nan
        if len(diff_points) == 0:
            df[('bathymetry', 'all')] = np.nan
            df[('benthic', 'all')] = np.nan
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

    @staticmethod
    def get_location_map_data(map_df, columns, location, crs):
        """
        Get the features of the columns at a certain location
        Parameters
        ----------
        map_df: DataFrame
            DataFrame where the geographical data to get the features from is
        columns : list of strings
            Columns to get the features from
        location : geometry object
            Location to get the features from
        crs : CRS projection
            Projection of the location

        Returns
        -------
        List of features in that particular location
        """
        project = pyproj.Transformer.from_crs(crs.geodetic_crs, map_df.crs, always_xy=True).transform
        location_crs = shapely.ops.transform(project, location)
        mask = map_df.contains(location_crs)
        try:
            idx = np.where(mask)[0][0]
            if isinstance(columns, list):
                features = map_df.loc[idx, columns].values
            else:
                features = map_df.loc[idx, columns]
        except IndexError:
            print('This location is not included in the map, setting to Nan')
            features = np.nan
        return features
