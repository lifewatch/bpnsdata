import geopandas
import pandas as pd
from importlib import resources


class CSVData:
    """

    """
    def __init__(self, csv_path, lon_col, lat_col, columns, suffix='csv'):
        self.csv_path = csv_path
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.columns = columns
        self.suffix = suffix

        csv_df = pd.read_csv(csv_path)
        self.df = geopandas.GeoDataFrame(csv_df, 
                                         geometry=geopandas.points_from_xy(x=csv_df[lon_col], 
                                                                           y=csv_df[lat_col]),
                                         crs='EPSG:4326')

    def __call__(self, df):
        return self.add_data_df(df)

    def add_data_df(self, df):
        """
        Add to each row the name of the closest shipwreck, its position in lat, lon degrees and the distance to it

        Parameters
        ----------
        df : GeoDataFrame
        """
        # Check for the nearest point (compute distance in Mercator projection!)
        df_dist = geopandas.sjoin_nearest(df[['geometry']].to_crs(epsg=3395),
                                          self.df.to_crs(epsg=3395),
                                          how='inner', distance_col='distance')
        df[self.suffix+'_distance'] = df_dist['distance']
        df[self.suffix+'_lat'] = df_dist[self.lat_col]
        df[self.suffix+'_lon'] = df_dist[self.lon_col]
        for col in self.columns:
            df[self.suffix+'_'+col] = df_dist[col]

        return df


class ShipwreckData(CSVData):
    def __init__(self):
        lat_col = 'features/geometry/coordinates/1'
        lon_col = 'features/geometry/coordinates/0'
        columns = ['features/properties/name']
        with resources.path('bpnsdata.data', 'wrakkendatabank.csv') as m:
            super().__init__(m, lon_col, lat_col, columns=columns, suffix='shipwreck')
