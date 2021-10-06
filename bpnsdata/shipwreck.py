import geopandas
import pandas as pd
from importlib import resources


# Adding shipwreck information
# The distance between the measurement and the closest shipwreck
# Coordinates and name of the closes shipwreck


class ShipWreck:
    def __init__(self):
        with resources.path('bpnsdata.data', 'wrakkendatabank.afdelingkust.be-json-export_2021-04-02.xls') as m:
            self.shipwrecks_path = m
            shipwrecks = pd.read_excel(self.shipwrecks_path, sheet_name='Sheet 1', header=0)
            self.shipwrecks = geopandas.GeoDataFrame(shipwrecks, geometry=geopandas.points_from_xy(
                                                     x=shipwrecks['features/geometry/coordinates/0'],
                                                     y=shipwrecks['features/geometry/coordinates/1']),
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
        df_dist = geopandas.sjoin_nearest(df[[('geom', '')]].droplevel(1, axis=1).to_crs(epsg=3395),
                                          self.shipwrecks.to_crs(epsg=3395),
                                          how='inner', distance_col='distance')
        df['shipwreck_distance'] = df_dist['distance']
        df['shipwreck_name'] = df_dist['features/properties/name']
        df['shipwreck_lat'] = df_dist['features/geometry/coordinates/1']
        df['shipwreck_longitude'] = df_dist['features/geometry/coordinates/0']
        return df
