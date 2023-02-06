import pandas as pd
import unittest
import numpy as np
import shapely
import geopandas

import bpnsdata

shapely.speedups.disable()


class TestSeaDataManager(unittest.TestCase):
    def setUp(self) -> None:
        # Create a random dataframe to work with
        time_index = pd.date_range(start='2020-10-12 11:36', end='2020-10-12 12:00', freq='min', tz='UTC')
        random_data = np.random.randint(5, 30, size=len(time_index))
        self.df = pd.DataFrame(random_data)
        self.df['datetime'] = time_index
        survey_location = bpnsdata.geolocation.SurveyLocation('test_data/VG.gpx')
        self.geodf = survey_location.add_location(self.df, datetime_column_df='datetime')
        # Set one point empty to test if the results are correctly given for empty points
        self.geodf.loc[self.geodf.iloc[0].name, 'geometry'] = shapely.geometry.Point()

    def test_sea_data_manager(self):
        env_vars = ['shipping', 'time', 'wrakken_bank', 'habitat_suitability',
                    'seabed_habitat', 'sea_surface', 'sea_wave', 'rain', 'wind', 'bathymetry']
        self.manager = bpnsdata.SeaDataManager(env_vars)
        df = self.manager(self.geodf, datetime_column='datetime', ais={'dt': '1min'}, verbose=True)
        print('All data')
        print(df)

    def test_geolocation(self):
        survey_location = bpnsdata.geolocation.SurveyLocation('test_data/VG.gpx')
        self.df = survey_location.add_location(self.df, time_tolerance='2.5s', other_cols=None)
        print('Geolocation')
        print(self.df)

    def test_rain(self):
        rain = bpnsdata.RainData()
        df = rain(self.geodf, datetime_column='datetime')
        print('Rain data')
        print(df)

    def test_wind(self):
        wind = bpnsdata.WindData()
        df = wind(self.geodf, datetime_column='datetime')
        print('Wind data')
        print(df)

    def test_shipping(self):
        shipping = bpnsdata.ShippingData()
        df = shipping(self.geodf, datetime_column='datetime')
        print('Shipping data (not ais)')
        print(df)

    def test_bathymetry(self):
        bathymetry = bpnsdata.BathymetryData()
        df = bathymetry(self.geodf, datetime_column='datetime')
        print('Bathymetry data')
        print(df)

    def test_shipwreck(self):
        shipwreck = bpnsdata.WrakkenBankData()
        df = shipwreck(self.geodf, datetime_column='datetime')
        print('Shipwrecks data')
        print(df)

    def test_sea_surface(self):
        sea_surface = bpnsdata.SeaSurfaceData()
        df = sea_surface(self.geodf, datetime_column='datetime')
        print('Sea surface data')
        print(df)

    def test_sea_wave(self):
        sea_wave = bpnsdata.SeaWaveData()
        df = sea_wave(self.geodf, datetime_column='datetime')
        print('Sea wave data')
        print(df)

    def test_seabed_habitat(self):
        seabed_habitat = bpnsdata.SeabedHabitatData()
        df = seabed_habitat(self.geodf, datetime_column='datetime')
        print('Seabed habitat data')
        print(df)

    def test_habitat_suitability(self):
        habitat_suitability = bpnsdata.HabitatSuitabilityData()
        df = habitat_suitability(self.geodf, datetime_column='datetime')
        print('Habitat suitability data')
        print(df)

    def test_time(self):
        time = bpnsdata.TimeData()
        df = time(self.geodf, datetime_column='datetime')
        print('Time data')
        print(df)

    def test_ais(self):
        lon, lat = [2.812594, 51.704417]
        time_index = pd.date_range(start='2017-08-21 11:35', end='2017-08-22 12:00', freq='min', tz='UTC')
        random_data = np.random.randint(5, 30, size=len(time_index))
        df = pd.DataFrame(random_data)
        df['datetime'] = time_index
        df['lat'] = lat
        df['lon'] = lon
        geodf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['lon'], df['lat']), crs='epsg:4326')
        ais = bpnsdata.AisData()
        df = ais(geodf, dt='1min')
        print('AIS data')
        print(df)


if __name__ == '__main__':
    unittest.main()
