import datetime

import geopandas
import numpy as np
import pandas as pd
import requests
import shapely
from tqdm.auto import tqdm
import erddapy


class SeaStateData:
    def __init__(self):
        """
        Sea State for BPNS data downloader
        """
        self.ph = RBINSerddap(dataset_id='BCZ_HydroState_V1',
                              columns=['surface_baroclinic_eastward_sea_water_velocity',
                                       'surface_baroclinic_northward_sea_water_velocity',
                                       'sea_surface_height_above_sea_level',
                                       'sea_surface_salinity',
                                       'sea_surface_temperature'])

        self.wv = RBINSerddap(dataset_id='WAM_ECMWF', columns=['hs', 'tm_1'])

    @staticmethod
    def set_spatial_bounds(df, e):
        """
        Return the minimum bounds from the gridapp which will return some data

        Parameters
        ----------
        df: DataFrame
            Evolution dataframe with datetime as index
        """
        min_lon, min_lat, max_lon, max_lat = df.total_bounds
        if abs(max_lon - min_lon) < e.delta_lon:
            max_lon += e.delta_lon
            min_lon -= e.delta_lon
        if abs(max_lat - min_lat) < e.delta_lat:
            max_lat += e.delta_lat
            min_lat -= e.delta_lat
        e.constraints['latitude>='] = min_lat
        e.constraints['latitude<='] = max_lat
        e.constraints['longitude>='] = min_lon
        e.constraints['longitude<='] = max_lon

    @staticmethod
    def set_temporal_bounds(df, e):
        """
        Assign to start_time and end_time attributes the minimum spatial time that will return some data

        Parameters
        ----------
        df: DataFrame
            Evolution dataframe with datetime as index
        """
        start_time = df.index.min()
        end_time = df.index.max()
        if (end_time - start_time) < e.delta_time:
            start_time -= e.delta_time
            end_time += e.delta_time
        e.constraints['time>='] = start_time
        e.constraints['time<='] = end_time

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
        seastate = self.get_griddap_df(df, self.ph)
        wavestate = self.get_griddap_df(df, self.wv)

        for col in self.ph.columns + self.wv.columns:
            df[(col, 'all')] = None
        df['surface_baroclinic_sea_water_velocity'] = np.nan

        if seastate is not None:
            time_indexes_s = pd.DataFrame(index=seastate.index.unique())
        else:
            time_indexes_s = None
        if wavestate is not None:
            time_indexes_w = pd.DataFrame(index=wavestate.index.unique())
        else:
            time_indexes_w = None

        moored = len(df.geometry.unique()) == 1
        closest_point_s = None
        closest_point_w = None
        for t, row in tqdm(df.iterrows(), total=len(df)):
            if row[('geometry', '')] is not None:
                if seastate is not None:
                    closest_point_s, closest_row_s = self._get_closest_row(seastate, row, t, time_indexes_s,
                                                                           closest_point_s)
                    df.loc[t, (self.ph.columns, 'all')] = closest_row_s[self.ph.columns].values[0]

                if wavestate is not None:
                    closest_point_w, closest_row_w = self._get_closest_row(wavestate, row, t, time_indexes_w,
                                                                           closest_point_w)
                    df.loc[t, (self.wv.columns, 'all')] = closest_row_w[self.wv.columns].values[0]

                if not moored:
                    closest_point_s, closest_point_w = None, None

        df['surface_baroclinic_sea_water_velocity'] = np.sqrt((df[['surface_baroclinic_eastward_sea_water_velocity',
                                                                   'surface_baroclinic_northward_sea_water_velocity'
                                                                   ]] ** 2).sum(axis=1))

        return df

    def get_griddap_df(self, df, e):
        self.set_temporal_bounds(df, e)
        self.set_spatial_bounds(df, e)

        try:
            griddap = e.to_pandas(response='csv', header=[0, 1]).dropna()
            griddap = griddap.droplevel(1, axis=1)
            griddap['time'] = pd.to_datetime(griddap.time)
            griddap.set_index('time', inplace=True)
            griddap = geopandas.GeoDataFrame(griddap,
                                             geometry=geopandas.points_from_xy(griddap['longitude'],
                                                                               griddap['latitude']),
                                             crs="EPSG:4326")
        except requests.exceptions.HTTPError:
            griddap = None
        return griddap

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
        square_wv = shapely.geometry.box(self.wv.min_lat,
                                         self.wv.min_lon,
                                         self.wv.max_lat,
                                         self.wv.max_lon)
        square_ph = shapely.geometry.box(self.ph.min_lat,
                                         self.ph.min_lon,
                                         self.ph.max_lat,
                                         self.ph.max_lon)
        return df.geometry.intersects(square_wv).all() and df.geometry.intersects(square_ph).all()

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
        df_t = state_df[state_df.index == selected_time]
        if closest_point is None:
            closest_point = shapely.ops.nearest_points(df_t.geometry.unary_union,
                                                       row[('geometry', '')])[0]
        closest_row = df_t[(df_t.latitude == closest_point.coords.xy[1]) &
                           (df_t.longitude == closest_point.coords.xy[0])]
        return closest_point, closest_row


class RBINSerddap(erddapy.ERDDAP):
    def __init__(self, dataset_id, columns):
        super().__init__(server='https://erddap.naturalsciences.be/erddap/', protocol='griddap')
        self.dataset_id = dataset_id
        self.griddap_initialize()
        self.columns = columns

        metadata = pd.read_csv(self.get_info_url(dataset_id, response='csv'))
        self.max_lat = float(metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL')
                                          & (metadata['Attribute Name'] == 'geospatial_lat_max'), 'Value'].values[0])
        self.min_lat = float(metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL')
                                          & (metadata['Attribute Name'] == 'geospatial_lat_min'), 'Value'].values[0])
        self.delta_lat = float(metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL')
                                            & (metadata['Attribute Name'] == 'geospatial_lat_resolution'),
                                            'Value'].values[0])
        self.max_lon = float(metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL')
                                          & (metadata['Attribute Name'] == 'geospatial_lon_max'), 'Value'].values[0])
        self.min_lon = float(metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL')
                                          & (metadata['Attribute Name'] == 'geospatial_lon_min'), 'Value'].values[0])
        self.delta_lon = float(metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL')
                                            & (metadata['Attribute Name'] == 'geospatial_lon_resolution'),
                                            'Value'].values[0])

        start_time = metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL') &
                                  (metadata['Attribute Name'] == 'time_coverage_start'),
                                  'Value'].values[0]
        self.start_time = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%SZ')
        end_time = metadata.loc[(metadata['Variable Name'] == 'NC_GLOBAL') &
                                (metadata['Attribute Name'] == 'time_coverage_end'),
                                'Value'].values[0]
        self.end_time = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%SZ')
        delta_time = datetime.datetime.strptime(metadata.loc[(metadata['Variable Name'] == 'time')
                                                             & (metadata['Row Type'] == 'dimension'),
                                                             'Value'].values[0].split('averageSpacing=')[-1],
                                                "%Hh %Mm %Ss")
        self.delta_time = datetime.timedelta(hours=delta_time.hour,
                                             minutes=delta_time.minute,
                                             seconds=delta_time.second)
