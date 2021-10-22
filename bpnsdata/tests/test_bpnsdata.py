import pandas as pd
import unittest
import numpy as np
import shapely

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

    def test_sea_data_manager(self):
        env_vars = ['shipping', 'time', 'wrakken_bank', 'habitat_suitability',
                    'seabed_habitat', 'sea_surface', 'sea_wave']
        self.manager = bpnsdata.SeaDataManager(env_vars)
        self.manager(self.geodf)

    def test_geolocation(self):
        survey_location = bpnsdata.geolocation.SurveyLocation('test_data/VG.gpx')
        self.df = survey_location.add_location(self.df)

    def test_shipping(self):
        shipping = bpnsdata.ShippingData()
        df = shipping(self.geodf)

    def test_bathymetry(self):
        bathymetry = bpnsdata.BathymetryData()
        df = bathymetry(self.geodf)

    def test_shipwreck(self):
        shipwreck = bpnsdata.WrakkenBankData()
        df = shipwreck(self.geodf)

    def test_sea_surface(self):
        sea_surface = bpnsdata.SeaSurfaceData()
        df = sea_surface(self.geodf)

    def test_sea_wave(self):
        sea_wave = bpnsdata.SeaWaveData()
        df = sea_wave(self.geodf)

    def test_seabed_habitat(self):
        seabed_habitat = bpnsdata.SeabedHabitatData()
        df = seabed_habitat(self.geodf)

    def test_habitat_suitability(self):
        habitat_suitability = bpnsdata.HabitatSuitabilityData()
        df = habitat_suitability(self.geodf)

    def test_time(self):
        time = bpnsdata.TimeData()
        df = time(self.geodf)


if __name__ == '__main__':
    unittest.main()
