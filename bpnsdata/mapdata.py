import pathlib

import geopandas
import numpy as np
import requests
from tqdm.auto import tqdm
from importlib import resources


class MapData:
    """
    Class to het the spatial data. Default is habitat type
    """

    def __init__(self, map_path=None, borders_name='Belgian', benthic_path=None):
        """

        Parameters
        ----------
        map_path: string or Path
            Path where the data is. If set to None the default is in VLIZ's archive (seabedhabitat)
        borders_name: string
            Country code (2 letters). Default if BE (Belgium)
        benthic_path: string or Path
            Path where the benthic habitat distribution is stored. If set to None, default is the ILVO map.
        """
        if map_path is None:
            with resources.path('bpnsdata.data', 'seabedhabitat_BE.shp') as m:
                self.map_path = m
        else:
            if not isinstance(map_path, pathlib.Path):
                self.map_path = pathlib.Path(map_path)

        if benthic_path is None:
            with resources.path('bpnsdata.data', 'Habitat Suitability2.shp') as m:
                self.benthic_path = m
        else:
            if not isinstance(benthic_path, pathlib.Path):
                self.benthic_path = pathlib.Path(benthic_path)

        self.map = geopandas.read_file(self.map_path)
        self.benthic = geopandas.read_file(self.benthic_path)
        self.borders_name = borders_name
        self.borders = self.read_borders()

    def __call__(self, df):
        return self.get_seabottom_data(df)

    def read_borders(self):
        query = 'https://geo.vliz.be/geoserver/wfs?request=getfeature&service=wfs&version=1.1.0&' \
                  'typename=MarineRegions:eez&outputformat=json&Filter=<PropertyIsEqualTo><PropertyName>' \
                  'geoname</PropertyName><Literal>%s Exclusive Economic Zone</Literal></PropertyIsEqualTo>' \
                % self.borders_name
        response = requests.get(query)
        if response.status_code == 200:
            borders = geopandas.read_file(response.text)
        else:
            borders = None

        return borders

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
        print('Adding seabed habitat and benthic communities')
        df_map = df[['geom']].sjoin(self.map.to_crs(df.crs)[columns + ['geometry']],
                                                         predicate='within')
        df_benthic = df[['geom']].sjoin(self.benthic.to_crs(df.crs)[['GRIDCODE', 'geometry']],
                                                             predicate='within')
        for col in columns:
            df[col] = df_map[col]
        df['benthic'] = df_benthic['GRIDCODE']

        print('Adding bathymetry...')
        # Change reference system to match the bathymetry requests
        df = df.to_crs(epsg=4326)
        diff_points = df.geometry.unique()
        for point in tqdm(diff_points, total=len(diff_points)):
            indxes = df[df.geometry == point].index
            df.loc[indxes, 'bathymetry'] = self.get_bathymetry(point)
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
            depth = np.nan
        return depth
