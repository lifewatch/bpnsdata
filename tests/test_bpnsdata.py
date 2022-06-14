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
        time_index = pd.date_range(start='2020-10-12 11:35', end='2020-10-12 12:00', freq='min', tz='UTC')
        random_data = np.random.randint(5, 30, size=len(time_index))
        self.df = pd.DataFrame(random_data, index=time_index)
        survey_location = bpnsdata.geolocation.SurveyLocation('test_data/VG.gpx')
        self.geodf = survey_location.geotrackpoints
        self.geodf.loc[self.geodf.iloc[0].name, 'geometry'] = shapely.geometry.Point()

    def test_sea_data_manager(self):
        env_vars = ['ais', 'shipping', 'time', 'wrakken_bank', 'habitat_suitability',
                    'seabed_habitat', 'sea_surface', 'sea_wave', 'rain', 'wind', 'bathymetry']
        self.manager = bpnsdata.SeaDataManager(env_vars)
        df = self.manager(self.geodf, ais={'dt': '1min'})
        print('All data')
        print(df)

    def test_geolocation(self):
        survey_location = bpnsdata.geolocation.SurveyLocation('test_data/VG.gpx')
        self.df = survey_location.add_location(self.df, time_tolerance='2.5s', other_cols=None)
        print('Geolocation')
        print(self.df)

    def test_rain(self):
        rain = bpnsdata.RainData()
        df = rain(self.geodf)
        print('Rain data')
        print(df)

    def test_wind(self):
        wind = bpnsdata.WindData()
        df = wind(self.geodf)
        print('Wind data')
        print(df)

    def test_shipping(self):
        shipping = bpnsdata.ShippingData()
        df = shipping(self.geodf)
        print('Shipping data (not ais)')
        print(df)

    def test_bathymetry(self):
        bathymetry = bpnsdata.BathymetryData()
        df = bathymetry(self.geodf)
        print('Bathymetry data')
        print(df)

    def test_shipwreck(self):
        shipwreck = bpnsdata.WrakkenBankData()
        df = shipwreck(self.geodf)
        print('Shipwrecks data')
        print(df)

    def test_sea_surface(self):
        sea_surface = bpnsdata.SeaSurfaceData()
        df = sea_surface(self.geodf)
        print('Sea surface data')
        print(df)

    def test_sea_wave(self):
        sea_wave = bpnsdata.SeaWaveData()
        df = sea_wave(self.geodf)
        print('Sea wave data')
        print(df)

    def test_seabed_habitat(self):
        seabed_habitat = bpnsdata.SeabedHabitatData()
        df = seabed_habitat(self.geodf)
        print('Seabed habitat data')
        print(df)

    def test_habitat_suitability(self):
        habitat_suitability = bpnsdata.HabitatSuitabilityData()
        df = habitat_suitability(self.geodf)
        print('Habitat suitability data')
        print(df)

    def test_time(self):
        time = bpnsdata.TimeData()
        df = time(self.geodf)
        print('Time data')
        print(df)

    def test_ais(self):
        lon, lat = [2.812594, 51.704417]
        time_index = pd.date_range(start='2017-08-21 11:35', end='2017-08-22 12:00', freq='min', tz='UTC')
        random_data = np.random.randint(5, 30, size=len(time_index))
        df = pd.DataFrame(random_data, index=time_index)
        df['lat'] = lat
        df['lon'] = lon
        geodf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['lon'], df['lat']), crs='epsg:4326')
        ais = bpnsdata.AisData()
        df = ais(geodf, dt='1min')
        print('AIS data')
        print(df)


if __name__ == '__main__':
    unittest.main()
