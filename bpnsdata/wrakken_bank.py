import geopandas
import pandas as pd
import requests


class WrakkenBankData:
    """

    """
    def __init__(self):
        url = 'https://wrakkendatabank.api.afdelingkust.be/v1/wrecks'
        response = requests.get(url)
        if response.status_code == 200:
            wrecks_json = response.json()['wrecks']
            df = pd.DataFrame(wrecks_json)
            df['x'] = None
            df['y'] = None
            for i, row in df.iterrows():
                df.loc[i, ['x', 'y']] = row['pseudoMercatorCoordinate']

        self.df = geopandas.GeoDataFrame(df,
                                         geometry=geopandas.points_from_xy(x=df['x'],
                                                                           y=df['y']),
                                         crs='EPSG:3857')
        geom_epsg4326 = self.df.geometry.to_crs(epsg=4326)
        self.df['latitude'] = geom_epsg4326.y
        self.df['longitude'] = geom_epsg4326.x

    def __call__(self, df, **kwargs):
        """
        Add the raster point to the df

        Parameters
        ----------
        df : GeoDataFrame
            datetime as index and a column geometry
        **kwargs: None
            Ignored, just for compatibility

        Returns
        -------
        The GeoDataFrame updated
        """
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
                                          how='left', distance_col='distance')

        if len(df_dist) != len(df):
            # Some rows are at the same distance from more than 1 shipwreck. We keep the first one.
            df_dist = df_dist[~df_dist.index.duplicated(keep='first')]
        df['shipwreck_distance'] = df_dist['distance']
        df['shipwreck_lat'] = df_dist['latitude']
        df['shipwreck_lon'] = df_dist['longitude']

        return df
