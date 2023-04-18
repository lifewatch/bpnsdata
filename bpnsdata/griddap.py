import datetime

import erddapy
import numpy as np
import pandas as pd
import warnings
import requests
import shapely.geometry as geom
import xarray as xr


class RBINSerddap(erddapy.ERDDAP):
    def __init__(self, dataset_id, columns):
        """
        Create an object like the ERDDAP but with the boundaries from RBINS datasets

        Parameters
        ----------
        dataset_id : str
            String representing the database
        columns : list of strings
            List of the columns from the dataset that are desired to acquire
        """
        super().__init__(server='https://erddap.naturalsciences.be/erddap/', protocol='griddap')
        self.dataset_id = dataset_id
        self.columns = columns
        self.griddap_initialize()

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

    def __call__(self, df):
        return self.get_data(df)

    def set_spatial_bounds(self, df):
        """
        Return the minimum bounds from the griddap which will return some test_data

        Parameters
        ----------
        df: GeoDataFrame
            Evolution dataframe with datetime as index
        """
        min_lon, min_lat, max_lon, max_lat = df.total_bounds
        if abs(max_lon - min_lon) < self.delta_lon:
            max_lon += self.delta_lon
            min_lon -= self.delta_lon
        if abs(max_lat - min_lat) < self.delta_lat:
            max_lat += self.delta_lat
            min_lat -= self.delta_lat
        self.constraints['latitude>='] = min_lat
        self.constraints['latitude<='] = max_lat
        self.constraints['longitude>='] = min_lon
        self.constraints['longitude<='] = max_lon

    def set_temporal_bounds(self, df):
        """
        Assign to start_time and end_time attributes the minimum spatial time that will return some test_data

        Parameters
        ----------
        df: GeoDataFrame
            Evolution dataframe with datetime as index
        """
        start_time = df.index.min()
        end_time = df.index.max()
        if (end_time - start_time) < self.delta_time:
            start_time -= self.delta_time
            end_time += self.delta_time
        self.constraints['time>='] = start_time
        self.constraints['time<='] = end_time

    def get_data(self, df):
        """
        Add the griddap test_data from e. It downloads the test_data from e with the df (temporal and spatial)
        constraints - max, min latitude and longitude and start and end time of the df -
        For all the columns of e (e.columns) it adds the corresponding value to each df row

        Parameters
        ----------
        df : GeoDataFrame
            Needs to have - at least - time as index (datetime) and a geometry column
        """
        self.set_temporal_bounds(df)
        self.set_spatial_bounds(df)

        warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)
        not_empty_values = ~(df.geometry.is_empty | df.geometry.isna())
        try:
            griddap = self.to_xarray(decode_times=True, decode_timedelta=False)
            # Is there a xarray function?
            lat_points = xr.DataArray(df.loc[not_empty_values].geometry.y, dims='points')
            lon_points = xr.DataArray(df.loc[not_empty_values].geometry.x, dims='points')
            time_points = xr.DataArray(df.loc[not_empty_values].index.values, dims='points')
            nearest_points = griddap.sel(latitude=lat_points, longitude=lon_points, time=time_points,
                                         method='nearest')
            for col in self.columns:
                df.loc[not_empty_values, col] = nearest_points[col].values
        except (requests.exceptions.HTTPError, PermissionError):
            df[self.columns] = np.nan

        return df

    def check_limits(self, df):
        """
        Check if all the points are within the available test_data range
        Parameters
        ----------
        df : GeoDataFrame
            DataFrame with all the rows to check
        Returns
        -------
        True if all the points are within the available range
        """
        square = geom.box(self.min_lat,
                          self.min_lon,
                          self.max_lat,
                          self.max_lon)
        return df.geometry.intersects(square).all()


class SeaSurfaceData(RBINSerddap):
    def __init__(self):
        """
        Sea State for BPNS test_data downloader
        """
        dataset_id = 'BCZ_HydroState_V1'
        columns = ['surface_baroclinic_eastward_sea_water_velocity',
                   'surface_baroclinic_northward_sea_water_velocity',
                   'sea_surface_height_above_sea_level',
                   'sea_surface_salinity',
                   'sea_surface_temperature']
        super().__init__(dataset_id, columns)

    def __call__(self, df):
        super().__call__(df)
        df['surface_baroclinic_sea_water_velocity'] = np.sqrt((df[['surface_baroclinic_eastward_sea_water_velocity',
                                                                   'surface_baroclinic_northward_sea_water_velocity'
                                                                   ]] ** 2).sum(axis=1))
        return df


class SeaBottomData(RBINSerddap):
    def __init__(self):
        """
        Sea State for BPNS test_data downloader
        """
        dataset_id = 'BCZ_HydroState_V1'
        columns = ['bottom_baroclinic_eastward_sea_water_velocity',
                   'bottom_baroclinic_northward_sea_water_velocity',
                   'bottom_upward_sea_water_velocity']
        super().__init__(dataset_id, columns)

    def __call__(self, df):
        super().__call__(df)
        df['bottom_baroclinic_sea_water_velocity'] = np.sqrt((df[['bottom_baroclinic_eastward_sea_water_velocity',
                                                                  'bottom_baroclinic_northward_sea_water_velocity'
                                                                  ]] ** 2).sum(axis=1))
        return df


class SeaWaveData(RBINSerddap):
    def __init__(self):
        dataset_id = 'WAM_ECMWF'
        columns = ['hs', 'tm_1']
        super().__init__(dataset_id, columns)
