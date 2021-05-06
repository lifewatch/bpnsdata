import datetime

import geopandas
import numpy as np
import pandas as pd
import requests
import shapely


class SeaStateData:
    def __init__(self):
        """
        Sea State for BPNS data downloader
        """
        self.seastate = pd.DataFrame()
        self.columns_ph = ['surface_baroclinic_eastward_sea_water_velocity',
                           'surface_baroclinic_northward_sea_water_velocity',
                           'sea_surface_height_above_sea_level',
                           'sea_surface_salinity',
                           'sea_surface_temperature']
        self.columns_wv = ['hs', 'tm_1']
        self.min_lat = 51.0
        self.max_lat = 51.91667
        self.min_lon = 2.083333
        self.max_lon = 4.214285

    def get_data(self, df):
        """
        Add all the sea state data to the df
        Parameters
        ----------
        df : DataFrame
            Evolution dataframe with datetime as index
        Returns
        -------
        The df with all the columns added
        """
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = df.total_bounds
        start_timestamp = (df.index.min() - datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_timestamp = (df.index.max() + datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        wavestate = self.get_griddap_df(start_timestamp, end_timestamp, 'WAM_ECMWF', self.columns_wv)
        seastate = self.get_griddap_df(start_timestamp, end_timestamp, 'BCZ_HydroState_V1', self.columns_ph)
        for col in self.columns_ph + self.columns_wv:
            df[(col, 'all')] = None

        if seastate is not None and wavestate is not None:
            time_indexes = pd.DataFrame(index=seastate.time.unique())
            moored = len(df.geometry.unique()) == 1
            closest_point_s = None
            closest_point_w = None
            for t, row in df.iterrows():
                if row[('geometry', '')] is not None:
                    closest_point_s, closest_row_s = self._get_closest_row(seastate, row, t, time_indexes,
                                                                           closest_point_s)
                    df.loc[t, (self.columns_ph, 'all')] = closest_row_s[self.columns_ph].values[0]

                    closest_point_w, closest_row_w = self._get_closest_row(wavestate, row, t, time_indexes,
                                                                           closest_point_w)
                    df.loc[t, (self.columns_wv, 'all')] = closest_row_w[self.columns_wv].values[0]
                    if not moored:
                        closest_point_s, closest_point_w = None, None
                else:
                    df.loc[t, (self.columns_wv, 'all')] = np.nan
            df['surface_baroclinic_sea_water_velocity'] = np.sqrt((df[['surface_baroclinic_eastward_sea_water_velocity',
                                                                       'surface_baroclinic_northward_sea_water_velocity'
                                                                       ]] ** 2).sum(axis=1))
        else:
            df[self.columns_wv] = np.nan
            df[self.columns_ph] = np.nan
            df['surface_baroclinic_sea_water_velocity'] = np.nan
        return df

    @staticmethod
    def _get_closest_row(state_df, row, t, time_indexes, closest_point=None):
        """
        Return the closest row from the state_df to the specified row and time. To avoid computing every time the
        closest point for a df with only one geometry point (but different times), pass the first closest point
        calculated

        Parameters
        ----------
        state_df: DataFrame
            DataFrame to get the closest row from
        row: DataFrame row
            Row of the current position to compute
        t: datetime
            Datetime of the row
        time_indexes: pandas Series
            Unique times of the state_df dataframe

        Returns
        -------
        closest_point (shapely point), closest_row (pandas row selected from state_df)
        """
        selected_time_idx = time_indexes.index.get_loc(t, method='nearest')
        selected_time = time_indexes.iloc[selected_time_idx].name
        df_t = state_df[state_df.time == selected_time]
        if closest_point is None:
            closest_point = shapely.ops.nearest_points(df_t.geometry.unary_union,
                                                       row[('geometry', '')])[0]
        closest_row = df_t[(df_t.latitude == closest_point.coords.xy[1]) &
                           (df_t.longitude == closest_point.coords.xy[0])]
        return closest_point, closest_row

    def get_griddap_df(self, start_timestamp, end_timestamp, table_name, columns):
        """
        Returns a DataFrame with the data from the griddap service between the specified dates, and the specified
        columns from the specified table
        Parameters
        ----------
        start_timestamp : datetime
            Start time to downlowad the data from
        end_timestamp : datetime
            End time to download the data from
        table_name : string
            Name of the table
        columns : list of strings
            List of all the columns to store

        Returns
        -------
        DataFrame with the downloaded table from griddap
        """
        query = 'https://erddap.naturalsciences.be/erddap/griddap/%s.json?' % table_name
        for col in columns:
            query = query + '%s[(%s):1:(%s)][(%s):1:(%s)][(%s):1:(%s)],' % (col, start_timestamp, end_timestamp,
                                                                            self.min_lat, self.max_lat,
                                                                            self.min_lon, self.max_lon)
        response = requests.get(query[:-1])
        if response.status_code == 200:
            griddap = pd.DataFrame(columns=response.json()['table']['columnNames'],
                                   data=response.json()['table']['rows'])
            griddap.dropna(inplace=True)
            griddap.time = pd.to_datetime(griddap.time)
            griddap = geopandas.GeoDataFrame(griddap,
                                             geometry=geopandas.points_from_xy(griddap.longitude, griddap.latitude),
                                             crs="EPSG:4326")
            return griddap
        else:
            return None

    def check_limits(self, df):
        """
        Check if all the points are within the available data range
        Parameters
        ----------
        df : GeopandasDataFrame
            DataFrame with all the rows to check
        Returns
        -------
        True if all the points are within the available range
        """
        square = shapely.geometry.box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)
        return df.geometry.intersects(square).all()
