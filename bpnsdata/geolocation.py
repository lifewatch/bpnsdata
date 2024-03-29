"""
Module: geolocation.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import pathlib
import sqlite3

import contextily as ctx
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry as sgeom
from geopy import distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
import requests
import numpy as np


class SurveyLocation:
    def __init__(self, geofile=None, datetime_column='datetime', **kwargs):
        """
        Location of all the survey points
        Parameters
        ----------
        geofile : str or Path
            Where the geo data is stored. can be a csv, gpx, pickle or sqlite3
        datetime_column: str
            Column where the datetime from geotrackpoints property will be stored
        """
        self.datetime_column = datetime_column
        if geofile is None:
            self.geotrackpoints = None
        else:
            if type(geofile) == str:
                geofile = pathlib.Path(geofile)
            extension = geofile.suffix
            if extension == '.gpx':
                geotrackpoints = self.read_gpx(geofile)
            elif extension == '.pkl':
                geotrackpoints = self.read_pickle(geofile, **kwargs)
            elif extension == '.csv':
                geotrackpoints = self.read_csv(geofile, **kwargs)
            elif extension == '.sqlite3':
                geotrackpoints = self.read_sqlite3(geofile, **kwargs)
            else:
                raise Exception('Extension %s is not implemented!' % extension)

            self.geotrackpoints = geotrackpoints
            self.geofile = geofile

    def read_gpx(self, geofile):
        """
        Read a GPX from GARMIN
        Parameters
        ----------
        geofile : str or Path
            GPX file
        Returns
        -------
        Geopandas DataFrame
        """
        geotrackpoints = geopandas.read_file(geofile)
        if len(geotrackpoints) == 0:
            geotrackpoints = geopandas.read_file(geofile, layer='track_points')

        geotrackpoints.drop_duplicates(subset='time', inplace=True)
        geotrackpoints[self.datetime_column] = pd.to_datetime(geotrackpoints['time'])
        geotrackpoints = geotrackpoints.sort_values(self.datetime_column)
        geotrackpoints.dropna(subset=['geometry'], inplace=True)

        return geotrackpoints

    def read_pickle(self, geofile, crs='EPSG:4326'):
        """
        Read a pickle file
        Parameters
        ----------
        geofile : string or path
            Where the gps information is
        crs : string
            Projection of the gps information

        Returns
        -------
        A Geopandas DataFrame
        """
        df = pd.read_pickle(geofile)
        geotrackpoints = geopandas.GeoDataFrame(df, crs=crs)
        geotrackpoints[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        geotrackpoints.drop_duplicates(subset=self.datetime_column, inplace=True)
        geotrackpoints = geotrackpoints.sort_index()
        geotrackpoints.dropna(subset=['geometry'], inplace=True)

        return geotrackpoints

    def read_csv(self, geofile, lat_col='Lat', lon_col='Lon'):
        """
        Read a csv file
        Parameters
        ----------
        geofile : string or path
            Where the gps information is
        lat_col : string
            Column with the Latitude
        lon_col : string
            Column with the Longitude test_data

        Returns
        -------
        A GeoPandasDataFrame
        """
        df = pd.read_csv(geofile)
        geotrackpoints = self.convert_df(df, self.datetime_column, lat_col, lon_col)
        return geotrackpoints

    def read_sqlite3(self, geofile, table_name='gpsData', lat_col='Latitude',
                     lon_col='Longitude'):
        """
        Read a sqlite3 file with geolocation test_data
        Parameters
        ----------
        geofile : str or Path
            sqlite3 file
        table_name : str
            name of the table to read from the sqlite3 db
        lat_col : string
            Column with the Latitude
        lon_col : string
            Column with the Longitude test_data

        Returns
        -------
        Geopandas DataFrame
        """
        conn = sqlite3.connect(geofile).cursor()
        query = conn.execute("SELECT * FROM %s" % table_name)
        cols = [column[0] for column in query.description]
        df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        geotrackpoints = self.convert_df(df, self.datetime_column, lat_col, lon_col)
        return geotrackpoints

    def convert_df(self, df, lat_col, lon_col):
        """
        Convert the DataFrame in a GeoPandas DataFrame
        Parameters
        ----------
        df : DataFrame
        datetime_col: str
            String where the time information is
        lat_col: str
            Column name where the latitude is
        lon_col: str
            Column name where the longitude is
        Returns
        -------
        The df turned in a geopandas dataframe
        """
        geotrackpoints = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[lon_col], df[lat_col]),
                                                crs='EPSG:4326')
        geotrackpoints[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        geotrackpoints.drop_duplicates(subset=self.datetime_column, inplace=True)
        geotrackpoints = geotrackpoints.sort_values(self.datetime_column)
        geotrackpoints.dropna(subset=[lat_col], inplace=True)
        geotrackpoints.dropna(subset=[lon_col], inplace=True)

        return geotrackpoints

    def add_location(self, df, datetime_column_df='datetime', time_tolerance='10min', other_cols=None):
        """
        Add the closest location of the GeoSurvey to each timestamp of the DataFrame.
        Returns a new GeoDataFrame with all the information of df but with added geometry

        Parameters
        ----------
        df : DataFrame
            DataFrame with a datetime column or an index datetime-like
        datetime_column_df: str
            Column where the datetime information is stored. Default to 'datetime'
        time_tolerance : str
            Maximum time to be acceptable distance between the survey timestamp and the geolocation point i.e. 10min
        other_cols : str

        """
        if other_cols is None:
            other_cols = []
        if self.geotrackpoints[self.datetime_column].dt.tz is None:
            self.geotrackpoints[self.datetime_column] = self.geotrackpoints[self.datetime_column].dt.tz_localize('UTC')

        try:
            if datetime_column_df in df.columns:
                df = df.sort_values(datetime_column_df)
                geo_df = pd.merge_asof(df, self.geotrackpoints[['geometry', self.datetime_column]+other_cols],
                                       left_on=datetime_column_df, right_on=self.datetime_column,
                                       tolerance=pd.Timedelta(time_tolerance), direction='nearest')
            else:
                # Check if the datetime is the index
                geo_df = pd.merge_asof(df.sort_index(), self.geotrackpoints[['geometry'] + other_cols],
                                       left_index=True, right_on=self.datetime_column,
                                       tolerance=pd.Timedelta(time_tolerance), direction='nearest')
        except ValueError:
            geo_df = df
            geo_df['geometry'] = np.nan

        df = geopandas.GeoDataFrame(geo_df, geometry='geometry', crs=self.geotrackpoints.crs.to_string())

        return df

    def plot_survey_color(self, column, df, map_file=None, save_path=None):
        """
        Add the closest location to each timestamp and color the points in a map

        Parameters
        ----------
        column : string
            Column of the df to plot
        df : DataFrame or GeoDataFrame
            DataFrame from an ASA output or GeoDataFrame
        map_file : string or Path
            Map that will be used as a basemap
        save_path : str or PathLike or file-like object
        """
        if 'geometry' not in df.columns:
            df = self.add_location(df)
        _, ax = plt.subplots(1, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if df[column].dtype != float:
            df.plot(column=column, ax=ax, legend=True, alpha=0.5, categorical=True, cax=cax)
        else:
            df.plot(column=column, ax=ax, legend=True, alpha=0.5, cmap='YlOrRd', categorical=False,
                    cax=cax)
        if map_file is None:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=ctx.providers.Stamen.TonerLite,
                            reset_extent=False)
        else:
            ctx.add_basemap(ax, crs=df.crs.to_string(), source=map_file, reset_extent=False,
                            cmap='BrBG')
        ax.set_axis_off()
        ax.set_title('%s Distribution' % column)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def add_distance_to(self, df, lat, lon, column='distance'):
        """
        Add the distances to a certain point.
        Returns GeoDataFrame with an added column with the distance to the point lat, lon

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            ASA output or GeoDataFrame
        lat : float
            Latitude of the point
        lon : float
            Longitude of the point
        column : str
            Name where to save the distance to
        """
        if "geometry" not in df.columns:
            df = self.add_location(df)
        df[column] = df['geometry'].apply(distance_m, lat=lat, lon=lon)

        return df

    def add_distance_to_coast(self, df, coastfile, column='coast_dist'):
        """
        Add the minimum distance to the coast.
        Returns GeoDataFrame with an added column with the distance to the coast

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            ASA output or GeoDataFrame
        coastfile : str or Path
            File with the points of the coast
        column : str
            Name where to save the distance to the coast
        """
        # TODO implement reading the coastline from emodnet bathymetry!
        if "geometry" not in df.columns:
            df = self.add_location(df)
        coastline = geopandas.read_file(coastfile, crs='EPSG:4326')
        coastline = coastline.to_crs('EPSG:3857').loc[0].geometry
        df[column] = df.to_crs('EPSG:3857').distance(coastline)

        return df


class Borders:
    def __init__(self, borders_name='Belgium'):
        """
        Parameters
        ----------
        borders_name: string
            Country code (2 letters). Default if Belgium
        """
        self.borders_name = borders_name

    def read_borders(self):
        query = 'https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&' \
                'request=GetFeature&typeName=eez_boundaries&outputformat=application/' \
                'json&Filter=<PropertyIsEqualTo><PropertyName>territory' \
                '</PropertyName><Literal>%s</Literal></PropertyIsEqualTo>' % self.borders_name
        response = requests.get(query)
        if response.status_code == 200:
            borders = geopandas.read_file(response.text)
        else:
            borders = None

        return borders


def distance_m(coords, lat, lon):
    """
    Return the distance in meters between the coordinates and the point (lat, lon)

    Parameters
    ----------
    coords: shapely point object
        First point
    lat: float
        Latitude of the second point
    lon: float
        Longitude of the second point
    """
    if coords is None:
        return None
    else:
        d = distance.distance((lat, lon), (coords.y, coords.x)).m
        return d


def min_distance_m(coords, geodf):
    """
    Return the minimum distance in meters between the coords and the points of the geodf
    Parameters
    ----------
    coords: shapely point object
        Coordinates to compute the min distance to
    geodf: GeoDataFrame
        The dataframe with the 'geometry' column to calculate the min distance to the point

    Returns
    -------
    The minimum distance between all the points of the geodf and the coords (in m, float)
    """
    if coords is None:
        return None
    else:
        distances = geodf['geometry'].apply(distance_m, args=(coords.y, coords.x))
        return distances.min()
