import urllib

import numpy as np
import rasterio
import requests
from tqdm import tqdm
import csv
import pandas as pd
from geopy import distance
from coordinates import Coordinate
import geopandas

# Adding shipwreck information
# The distance between the measurement and the closest shipwreck
# Coordinates and name of the closes shipwreck

class ShipWreck:
    def __init__(self):
        self.wrakkenfile = '//fs/SHARED/onderzoek/6. Marine Observation Center/Projects/PhD_Clea/Data/wrakkendatabank/wrakkendatabank.afdelingkust.be-json-export_2021-04-02.xls'


    def get_shipwreck_information_df(self, df):
        # Add reading to init
        wrakkenfile = self.wrakkenfile
        dfs = pd.read_excel(wrakkenfile, sheet_name='Sheet 1', header=0)
        geotrackpoints = geopandas.GeoDataFrame(dfs, geometry=geopandas.points_from_xy(x=dfs['features/geometry/coordinates/0'],y=dfs['features/geometry/coordinates/1']),
                                                crs='EPSG:4326')
        df[('shipwrecks','distance')] = None
        df[('shipwrecks','co_x')] = None
        df[('shipwrecks','co_y')] = None
        df[('shipwrecks','name_ship')] = None
        try:
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                if row['geometry', ''] is not None:
                    dis = min_distance_m(row['geometry', ''], geotrackpoints)

                    df.loc[idx,('shipwrecks','distance')] = dis[0]
                    df.loc[idx,('shipwrecks','co_x')] = dfs['features/geometry/coordinates/0'][dis[1]]
                    df.loc[idx,('shipwrecks','co_y')] = dfs['features/geometry/coordinates/1'][dis[1]]
                    df.loc[idx,('shipwrecks','name_ship')] = dfs['features/properties/name'][dis[1]]
            
            # df['geometry']

        except AttributeError:
            print('An attribute is missing to complete shipwrecks')
            df.loc[idx,('shipwrecks','distance')] = np.nan
            df.loc[idx,('shipwrecks','co_x')] = np.nan
            df.loc[idx,('shipwrecks','co_y')] = np.nan
            df.loc[idx,('shipwrecks','name_ship')] = np.nan
        return df

        

def distance_m(coords, lat, lon):
    """
    Return the distance in meters between the coordinates and the point (lat, lon)
    """
    if coords is None:
        return None
    else:
        d = distance.distance((lat, lon), (coords.y, coords.x)).m
        return d


def min_distance_m(coords, geodf):
    """
    Return the minimum distance in meters between the coords and the points of the geodf
    """
    if coords is None:
        return None
    else:
        distances = geodf['geometry'].apply(distance_m, args=(coords.y, coords.x))
        return [distances.min(), distances.argmin()]
