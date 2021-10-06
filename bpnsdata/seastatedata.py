import datetime

import erddapy
import numpy as np
import pandas as pd
import requests
import shapely.geometry as geom
import xarray as xr


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

    def __call__(self, df):
        return self.get_data(df)

    @staticmethod
    def set_spatial_bounds(df, erddap_e):
        """
        Return the minimum bounds from the griddap which will return some data

        Parameters
        ----------
        df: GeoDataFrame
            Evolution dataframe with datetime as index
        erddap_e: RBINSerddap object
            Entry point for the erddap dataset
        """
        min_lon, min_lat, max_lon, max_lat = df.total_bounds
        if abs(max_lon - min_lon) < erddap_e.delta_lon:
            max_lon += erddap_e.delta_lon
            min_lon -= erddap_e.delta_lon
        if abs(max_lat - min_lat) < erddap_e.delta_lat:
            max_lat += erddap_e.delta_lat
            min_lat -= erddap_e.delta_lat
        erddap_e.constraints['latitude>='] = min_lat
        erddap_e.constraints['latitude<='] = max_lat
        erddap_e.constraints['longitude>='] = min_lon
        erddap_e.constraints['longitude<='] = max_lon

    @staticmethod
    def set_temporal_bounds(df, erddap_e):
        """
        Assign to start_time and end_time attributes the minimum spatial time that will return some data

        Parameters
        ----------
        df: GeoDataFrame
            Evolution dataframe with datetime as index
        erddap_e: RBINSerddap object
            Entry point for the erddap dataset
        """
        start_time = df.index.min()
        end_time = df.index.max()
        if (end_time - start_time) < erddap_e.delta_time:
            start_time -= erddap_e.delta_time
            end_time += erddap_e.delta_time
        erddap_e.constraints['time>='] = start_time
        erddap_e.constraints['time<='] = end_time

    def get_data(self, df):
        """
        Add all the sea state data to the df
        Parameters
        ----------
        df : GeoDataFrame
            Evolution dataframe with datetime as index
        Returns
        -------
        The df with all the columns added
        """
        df = self.add_griddap_data_to_df(df, self.ph)
        df = self.add_griddap_data_to_df(df, self.wv)
        df['surface_baroclinic_sea_water_velocity'] = np.sqrt((df[['surface_baroclinic_eastward_sea_water_velocity',
                                                                   'surface_baroclinic_northward_sea_water_velocity'
                                                                   ]] ** 2).sum(axis=1))
        return df

    def add_griddap_data_to_df(self, df, erddap_e):
        """
        Add the griddap data from e. It downloads the data from e with the df (temporal and spatial) constraints
        - max, min latitude and longitude and start and end time of the df -
        For all the columns of e (e.columns) it adds the corresponding value to each df row

        Parameters
        ----------
        df : GeoDataFrame
            Needs to have - at least - time as index (datetime) and a geometry column
        erddap_e : RBINSerddap object
            Entry point for the erddap dataset
        """
        self.set_temporal_bounds(df, erddap_e)
        self.set_spatial_bounds(df, erddap_e)

        try:
            griddap = erddap_e.to_xarray(decode_times=True)
            points_y = xr.DataArray(df.geometry.y)
            points_x = xr.DataArray(df.geometry.x)
            points_time = xr.DataArray(df.index.values)
            nearest_points = griddap.sel(latitude=points_y, longitude=points_x,
                                         time=points_time, method='nearest')[erddap_e.columns].to_dataframe()
            df[erddap_e.columns] = nearest_points
        except requests.exceptions.HTTPError:
            df[erddap_e.columns] = np.nan

        return df

    def check_limits(self, df):
        """
        Check if all the points are within the available data range
        Parameters
        ----------
        df : GeoDataFrame
            DataFrame with all the rows to check
        Returns
        -------
        True if all the points are within the available range
        """
        square_wv = geom.box(self.wv.min_lat,
                             self.wv.min_lon,
                             self.wv.max_lat,
                             self.wv.max_lon)
        square_ph = geom.box(self.ph.min_lat,
                             self.ph.min_lon,
                             self.ph.max_lat,
                             self.ph.max_lon)
        return df.geometry.intersects(square_wv).all() and df.geometry.intersects(square_ph).all()


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
