import rasterio
import urllib
import xarray as xr
import numpy as np
from tqdm import tqdm
import requests
import pandas as pd
import warnings
import shapely


class AisData:
    def __init__(self, buffer=10000, ):
        """
        Will return the cumulative shipping in a certain radius

        Parameters
        ----------
        buffer: float
            Radius to consider, in meters
        """
        self.url = 'http://aisdb.vliz.be:9000/functions/acoustic_impact/'
        self.buffer = buffer
        self.column_name = None
        self.limit = 10000

    def __call__(self, df, dt):
        """
        Add the specified ais data to the df

        Parameters
        ----------
        df : GeoDataFrame
            datetime as index and a column geometry

        Returns
        -------
        The GeoDataFrame updated
        """
        df_4326 = df.to_crs(epsg=4326)
        warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)
        not_empty_values = ~(df_4326.geometry.is_empty | df_4326.geometry.isna())
        centroid = shapely.geometry.LineString([xy for xy in df_4326.loc[not_empty_values, 'geometry']]).centroid
        df = self.get_all_ships_onepoint(df, centroid, dt)

        return df

    def get_all_ships_onepoint(self, df, point, dt='5s'):
        """

        Parameters
        ----------
        df : GeoDataFrame
            With datetime as index and geometry as a column
        point : Shapely Point
            Shapely point object

        Returns
        -------

        """
        dt = pd.to_timedelta(dt)
        lon, lat = point.coords[0]
        start_time_str = df.index.min().strftime('%Y-%m-%d %H:%M')
        end_time_str = df.index.max().strftime('%Y-%m-%d %H:%M')
        try:
            response = requests.get(self.url + '/items.json?', {'limit': self.limit,
                                                                'lon': str(lon),
                                                                'lat': str(lat),
                                                                'buffer_meters': self.buffer,
                                                                'start_time': start_time_str,
                                                                'end_time': end_time_str})
            df['ais_total_seconds'] = 0
            df['ais_n_ships'] = 0
            df['ais_total_seconds_distance_weighted'] = 0
            if response.status_code == 200:
                ais_data = response.json()
                ais_df = pd.DataFrame(columns=['distance_to_center', 'start_segment', 'end_segment'])
                if ais_data['numberReturned'] > 0:
                    for json_entry in ais_data['features']:
                        ais_df.loc[len(ais_df)] = [json_entry['properties']['Distance to Center'],
                                                   json_entry['properties']['Segment Start Time'],
                                                   json_entry['properties']['Segment End Time']]
                    ais_df['start_segment'] = pd.to_datetime(ais_df['start_segment'])
                    ais_df['end_segment'] = pd.to_datetime(ais_df['end_segment'])

                    for start_time, row in tqdm(df.iterrows(), total=len(df)):
                        end_time = start_time + dt
                        ais_df['start_segment_in_bin'] = np.maximum(ais_df['start_segment'], start_time)
                        ais_df['end_segment_in_bin'] = np.minimum(ais_df['end_segment'], end_time)
                        ais_df['duration'] = ais_df['end_segment_in_bin'] - ais_df['start_segment_in_bin']
                        ais_df['total_seconds'] = ais_df['duration'].dt.total_seconds()
                        mask = ais_df.total_seconds > 0
                        n_ships = mask.sum()
                        if n_ships > 0:
                            total_seconds = ais_df.loc[mask].total_seconds.sum()
                            total_seconds_weighted = (ais_df.loc[mask].total_seconds *
                                                      ais_df.loc[mask].distance_to_center).sum() / total_seconds

                            df.loc[start_time, ['ais_total_seconds',
                                                'ais_n_ships',
                                                'ais_total_seconds_distance_weighted']] = [total_seconds,
                                                                                           n_ships,
                                                                                           total_seconds_weighted]

            else:
                print('There is no data for these days or the request exceeded the maximum time. Please try again '
                      'with a shorter temporal request')

        except requests.exceptions.ConnectionError:
            print('Connection with the AIS database could not be done. Setting data to nan...')
            df['ais_total_seconds'] = np.nan
            df['ais_total_seconds_distance_weighted'] = np.nan
            df['ais_n_ships'] = np.nan

        return df
