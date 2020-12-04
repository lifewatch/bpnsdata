import datetime
import pathlib

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely
from pypam import acoustic_survey, geolocation
from skyfield import almanac, api


class DataSet:
    def __init__(self, summary_path, output_folder, instruments, features, bands_list=None, binsize=60.0, nfft=512):
        self.metadata = pd.read_csv(summary_path)
        self.instruments = instruments
        self.features = features
        self.bands_list = bands_list
        self.binsize = binsize
        self.nfft = nfft
        self.deployments = []
        self.dataset = pd.DataFrame()
        if not isinstance(output_folder, pathlib.Path):
            output_folder = pathlib.Path(output_folder)
        self.output_folder = output_folder
        self.output_folder.joinpath('img/temporal_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/data_overview').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/distance_vs_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/temporal_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('deployments').mkdir(parents=True, exist_ok=True)

    def generate_entire_dataset(self):
        for index in self.metadata.index:
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.sensitivity = deployment_row['hydrophone_sensitivity']
            d = Deployment(hydrophone=inst,
                           location=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'])
            d.generate_deployment_data(self.features, self.bands_list, self.binsize, self.nfft)
            d.add_spatial_data()
            d.evo.to_pickle(self.output_folder.joinpath('deployments/%s_%s.pkl' % (index, deployment_row['location'])))
            self.deployments.append(d)
            self.dataset.append(d.evo)
        self.dataset.to_pickle(self.output_folder.joinpath('dataset.pkl'))

    def add_metadata(self):
        """
        Return a db with a data overview of the folder
        """
        metadata_params = ['start_datetime', 'end_datatime', 'duration']
        metadata = pd.DataFrame(columns=list(self.metadata.columns) + metadata_params)
        for index in self.metadata.index:
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.sensitivity = deployment_row['hydrophone_sensitivity']
            d = Deployment(hydrophone=inst,
                           station_name=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'])
            metadata.at[index, ['start_datetime', 'end_datetime', 'duration']] = d.get_metadata()
        return metadata

    def plot_distr_over(self, map_path, map_column=None, deployment_column='geometry'):
        map = geopandas.read_file(map_path)
        ax = map.plot(column=map_column)
        self.evo.to_crs(map.crs).plot(column=deployment_column, ax=ax)
        plt.show()


class Deployment:
    def __init__(self, hydrophone, location, method, hydrophone_depth, data_folder_path, gps_path):
        self.hydrophone = hydrophone
        self.station_name = location
        self.method = method
        self.hydrophone_depth = hydrophone_depth
        self.data_folder_path = data_folder_path
        self.gps_path = gps_path
        self.geoloc = geolocation.SurveyLocation(self.gps_path)
        self.evo = None

        self.mapdata = MapData()
        self.timedata = TimeData()

    def __getattr__(self, item):
        if item == 'evo':
            if self.__dict__[item] is None:
                raise BrokenPipeError('You should first generate the deployment data! '
                                      'You can use generate_deployment_data for that')
        return self.__dict__[item]

    def get_metadata(self):
        """
        Get all the metadata of the location
        """
        asa = acoustic_survey.ASA(hydrophone=self.hydrophone, folder_path=self.data_folder_path)
        start, end = asa.start_end_timestamp()
        duration = asa.duration()
        return start, end, duration

    def generate_deployment_data(self, features=None, bands_list=None, binsize=60.0, nfft=512):
        """
        Generate a dataset in a pandas data frame with the spl values
        """
        asa = acoustic_survey.ASA(self.hydrophone, self.data_folder_path, binsize=binsize, nfft=nfft)
        if features is not None:
            evo = asa.evolution_multiple(method_list=features,
                                         band_list=bands_list)
        else:
            if self.method == 'Moored':
                start, _ = asa.start_end_timestamp()
                evo = pd.DataFrame({'datetime': start}, index=[0])
            else:
                evo = asa.timestamps_df()
        self.evo = evo
        return evo

    def add_gps_data(self):
        evo_loc = self.geoloc.add_survey_location(self.evo)
        self.evo = evo_loc
        return self.evo

    def add_spatial_features(self, lat=None, lon=None, column=None, coastfile=None):
        # Read the gpx (time is in UTC)
        datetime_df = pd.DataFrame({'datetime': self.evo.index.values})
        if lat is not None and lon is not None:
            evo_loc = self.geoloc.add_distance_to(df=datetime_df,
                                                  lat=lat,
                                                  lon=lon,
                                                  column=column)
        if coastfile is not None:
            evo_loc = self.geoloc.add_distance_to_coast(df=evo_loc, coastfile=coastfile, column='coast_dist')
        evo_loc = evo_loc.set_index('datetime')
        evo_loc.columns = pd.MultiIndex.from_product([evo_loc.columns, ['all']])
        evo = self.evo.join(evo_loc)
        self.evo = evo
        return self.evo

    def plot_features_vs_distance_to(self, features, geo_column):
        """
        Generate a dataset in a pandas data frame with the spl values
        """
        for feature in features:
            plt.figure()
            plt.scatter(x=self.evo[geo_column], y=self.evo['spl_lf'], label='LF')
            plt.xlabel('Distance [m]')
            plt.ylabel('SPL [dB]')
            plt.title('%s vs distance to %s in %s' % (feature, geo_column, self.station_name))
            plt.ylim(ymax=140, ymin=60)
            plt.legend()
            plt.close()

    def save(self, folder_path):
        self.evo.to_pickle(folder_path.joinpath(self.station_name + '.pkl'))


class TimeData:
    def __init__(self):
        self.ts = api.load.timescale()
        self.eph = api.load_file('acoustic_data_exploration/data/de421.bsp')

    def get_moon_phase(self, dt):
        """
        Return the moon phase of a certain date
        Parameters
        ----------
        dt: datetime object
            Datetime on which to calculate the moon phase
        Returns
        -------
        Moon phase as string
        """
        utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
        t = self.ts.utc(utc_dt)
        moon_phase_at = almanac.moon_phases(self.eph)
        moon_phase = moon_phase_at(t)
        return almanac.MOON_PHASES[moon_phase]

    def get_day_moment(self, dt, location):
        bluffton = api.Topos(latitude_degrees=location.coords[0][0], longitude_degrees=location.coords[0][1])
        utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
        t = self.ts.utc(utc_dt)
        is_dark_twilight_at = almanac.dark_twilight_day(self.eph, bluffton)
        day_moment = is_dark_twilight_at(t).min()
        return almanac.TWILIGHTS[day_moment]


class MapData:
    def __init__(self):
        self.wgs84 = pyproj.CRS('EPSG:4326')

    def get_location_map_data(self, map_path, column, location):
        if not isinstance(map_path, pathlib.Path):
            map_path = pathlib.Path(map_path)
        map = geopandas.read_file(map_path)
        location_crs = shapely.ops.transform(self.transform, location)
        mask = self.habitat_map.contains(location_crs)
        idx = np.where(mask)[0][0]
        feature = map.loc[idx, column]
        return feature

    def get_df_map_data(self):
        pass

    @staticmethod
    def get_bathymetry(location):
        point_str = str(location)
        response = requests.get("https://rest.emodnet-bathymetry.eu/depth_sample?geom=%s" % point_str)
        if response.status_code == 200:
            depth = response.json()['avg']
        else:
            depth = 0.0
        return depth
