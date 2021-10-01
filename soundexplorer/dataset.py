import pathlib

import contextily as ctx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypam import acoustic_survey, geolocation
from soundexplorer import mapdata, timedata, seastatedata, humandata, shipwreck
from tqdm import tqdm


shapely.speedups.disable()


class DataSet:
    def __init__(self, summary_path, output_folder, instruments, features, third_octaves=None,
                 bands_list=None, binsize=60.0, nfft=512):
        """
        A DataSet object is a representation of a group of acoustic deployments.
        It allows to calculate all the acoustic features from all the deployments and store them in a structured way
        in the output folder. The structure is as follows:
        output_folder
          - deployments : a pkl file for each deployment
          - detections : files resulting of the events detections. One folder per detection type
          - img : graphs and figures
            - temporal_features : temporal evolution of all features per deployment
            - data_overview : spatial and temporal coverage, and methods used
            - features_analysis : ??
            - spatial_features : spatial distribution of features
          dataset.pkl : pkl with all the datasets together
        Parameters
        ----------
        summary_path : string or Path
            Path to the csv file where all the metadata of the deployments is
        output_folder : string or Path
            Where to save the output files (pkl) of the deployments with the processed data
        instruments : dictionary of (name,  hydrophone_object) entries
            A dictionary of all the instruments used in the deployments
        features : list of strings
            A list of all the features to be calculated
        bands_list : list of tuples
            A list of all the bands to consider (low_freq, high_freq)
        third_octaves : False or band
            If False, no octave bands are calculated. Otherwise the parameter is passed to the pypam as a band
        binsize : float
            In seconds, duration of windows to consider
        nfft : int
            Number of samples of window to use for frequency analysis
        """
        self.metadata = pd.read_csv(summary_path)
        self.summary_path = summary_path
        self.instruments = instruments
        self.features = features
        self.third_octaves = third_octaves
        self.band_list = bands_list
        self.binsize = binsize
        self.nfft = nfft

        if not isinstance(output_folder, pathlib.Path):
            output_folder = pathlib.Path(output_folder)
        self.output_folder = output_folder
        self.output_folder.joinpath('img/temporal_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/data_overview').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/features_analysis').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/spatial_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('deployments').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('detections').mkdir(parents=True, exist_ok=True)

        already_computed_deployments = list(self.output_folder.joinpath('deployments').glob("*.pkl"))
        if len(already_computed_deployments) > 0:
            self.deployments = already_computed_deployments
        else:
            self.deployments = []
        self.dataset = None
        self.read_dataset()

    def generate_entire_dataset(self, coastfile=None, env_vars=None):
        """
        Calculates the acoustic features of every deployment and saves them as a pickle in the deployments folder with
        the name of the station of the deployment.
        Also adds all the deployment data to the self object in the general dataset,
        and the path to each deployment's pickle in the list of deployments
        """
        if env_vars is None:
            env_vars = []
        for index in tqdm(self.metadata.index, total=len(self.metadata)):
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.Vpp = deployment_row['hydrophone_Vpp']
            if deployment_row['instrument'] == 'B&K':
                inst.amplif = deployment_row['hydrophone_amp'] / 1e3
            d = Deployment(hydrophone=inst,
                           station_name=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'],
                           utc=deployment_row['utc'],
                           include_dirs=bool(deployment_row['include_dirs']),
                           etn_id=deployment_row['etn_id'])

            deployment_path = self.output_folder.joinpath('deployments/%s_%s.pkl' % (index, d.station_name))
            if deployment_path.exists():
                d.evo = pd.read_pickle(deployment_path)
                d.evo.drop_duplicates(inplace=True)
                if 'geometry' in d.evo.columns:
                    d.evo = geopandas.GeoDataFrame(d.evo, geometry='geom', crs='EPSG:4326')
                else:
                    d.add_spatial_data()
                d.evo.to_pickle(deployment_path)
            else:
                d.generate_deployment_data(features=self.features, band_list=self.band_list,
                                           third_octaves=self.third_octaves, binsize=self.binsize, nfft=self.nfft)
                # Update the metadata
                self.metadata.loc[index, 'hydrophone_sensitivity'] = d.hydrophone.sensitivity
                self.metadata.loc[index, 'hydrophone_Vpp'] = d.hydrophone.Vpp

                if 'spatial_data' in env_vars:
                    print('Adding spatial data...')
                    d.add_spatial_data()
                    if coastfile is not None:
                        d.add_distance_to_coast(coastfile=coastfile)
                if 'sea_state' in env_vars:
                    print('Adding seastate information...')
                    d.add_seastate()
                if 'time_data' in env_vars:
                    print('Adding Time Data information...')
                    d.add_time_data()
                if 'sea_bottom' in env_vars:
                    print('Adding Sea Bottom information...')
                    d.add_seabottom_data()
                if 'shipping' in env_vars:
                    print('Adding shipping information...')
                    d.add_shipping_data()
                if 'shipwreck' in env_vars:
                    print('Adding shipwreck information...')
                    d.add_shipwreck_info()
                d.evo.to_pickle(deployment_path)
                self.deployments.append(deployment_path)
            self.dataset = self.dataset.append(d.evo)
        self.dataset.to_pickle(self.output_folder.joinpath('dataset.pkl'))
        self.metadata.to_csv(self.summary_path, index=False)

    def read_dataset(self):
        """
        Read back all the generated dataset in case it was already generated
        """
        dataset_path = self.output_folder.joinpath('dataset.pkl')
        if dataset_path.exists():
            df = pd.read_pickle(dataset_path)
            if 'geometry' in df.columns:
                self.dataset = geopandas.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            else:
                self.dataset = df
        else:
            self.dataset = pd.DataFrame()

    def read_all_deployments(self):
        """
        Read all the deployments in case they were already generated (but dataset was not generated or not complete)
        """
        self.dataset = geopandas.GeoDataFrame()

        for index, deployment_file in enumerate(self.deployments):
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.sensitivity = deployment_row['hydrophone_sensitivity']
            d = Deployment(hydrophone=inst,
                           station_name=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'],
                           utc=deployment_row['utc'],
                           include_dirs=bool(deployment_row['include_dirs']),
                           etn_id=deployment_row['etn_id'])
            d.evo = pd.read_pickle(deployment_file)
            d.evo.drop_duplicates(inplace=True)
            if 'geometry' in d.evo.columns:
                d.evo = geopandas.GeoDataFrame(d.evo, geometry='geom', crs='EPSG:4326')
            else:
                print('Adding spatial data...')
                d.add_spatial_data()
            self.dataset = self.dataset.append(d.evo)
        self.dataset.set_geometry('geom', crs='EPSG:4326', inplace=True)
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
                           gps_path=deployment_row['gps_path'],
                           utc=deployment_row['utc'],
                           include_dirs=bool(deployment_row['include_dirs']),
                           etn_id=deployment_row['etn_id'])
            metadata.at[index, ['start_datetime', 'end_datetime', 'duration']] = d.get_metadata()
        return metadata

    def add_env_vars(self, env_vars):
        self.dataset = geopandas.GeoDataFrame()

        for index, deployment_file in enumerate(self.deployments):
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.sensitivity = deployment_row['hydrophone_sensitivity']
            d = Deployment(hydrophone=inst,
                           station_name=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'],
                           utc=deployment_row['utc'],
                           include_dirs=bool(deployment_row['include_dirs']),
                           etn_id=deployment_row['etn_id'])
            d.evo = pd.read_pickle(deployment_file)
            d.evo.drop_duplicates(inplace=True)
            if 'geometry' in d.evo.columns:
                d.evo = geopandas.GeoDataFrame(d.evo, geometry='geom', crs='EPSG:4326')
            else:
                print('Adding spatial data...')
                d.add_spatial_data()

            if 'spatial_data' in env_vars:
                print('Adding spatial data...')
                d.add_spatial_data()
            if 'sea_state' in env_vars:
                print('Adding seastate information...')
                d.add_seastate()
            if 'time_data' in env_vars:
                print('Adding Time Data information...')
                d.add_time_data()
            if 'sea_bottom' in env_vars:
                print('Adding Sea Bottom information...')
                d.add_seabottom_data()
            if 'shipping' in env_vars:
                print('Adding shipping information...')
                d.add_shipping_data()
            if 'shipwreck' in env_vars:
                print('Adding shipwreck information...')
                d.add_shipwreck_info()
            self.dataset = self.dataset.append(d.evo)

        self.dataset.set_geometry('geom', crs='EPSG:4326', inplace=True)
        self.dataset.to_pickle(self.output_folder.joinpath('dataset.pkl'))


    def plot_distr(self, column, band=None, map_file=None, map_column=None, categorical=False, borders=False):
        """
        Generate the distribution figures for the specific column for the whole dataset
        Parameters
        ----------
        column : string
            Column to plot the distribution of
        band : int
            Band to plot the distribution of. If None, will be chosen 'all'
        map_file : string or Path
            background map to use
        map_column : string
            Column of the map to plot. Default is None and will plot the geometry points
        categorical : bool
            Set to True if it is a categorical variable
        borders : bool
            Set to True if it is desired to plot the borders from the mapdata
        """
        if band is None:
            band = 'all'
            band_name = 'Broadband'
        else:
            band_name = self.band_list[band]
        if isinstance(map_file, str):
            map_file = pathlib.Path(map_file)

        # Clean the dataset for easier plotting
        clean_ds = self.dataset.iloc[:, self.dataset.columns.get_level_values('band') == band]
        clean_ds.columns = clean_ds.columns.droplevel('band')
        clean_ds = clean_ds.set_geometry(self.dataset.geometry)
        clean_ds = clean_ds.drop_duplicates(subset=['geom', column])
        clean_ds = clean_ds.dropna(axis=0, how='any', subset=['geom'])

        _, ax = plt.subplots(1, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if categorical:
            clean_ds.to_crs('EPSG:3857').plot(column=column, ax=ax, legend=True, alpha=0.5, categorical=True, zorder=3,
                                              cmap='YlOrRd')
        else:
            clean_ds[column] = clean_ds[column].astype(float)
            clean_ds.to_crs('EPSG:3857').plot(column=column, ax=ax, legend=True, alpha=0.5, cmap='YlOrRd',
                                              categorical=False,
                                              cax=cax, zorder=3)
        if borders:
            mapd = mapdata.MapData()
            mapd.borders.to_crs('EPSG:3857').plot(ax=ax, legend=False, zorder=2)
        if map_file is None:
            ctx.add_basemap(ax, reset_extent=False)
        else:
            if map_file.suffix == '.shp':
                map_df = geopandas.read_file(map_file)
                map_df.to_crs('EPSG:3857').plot(column=map_column, alpha=0.5, legend=True, ax=ax, zorder=1)
            else:
                ctx.add_basemap(ax, source=map_file, reset_extent=False, cmap='BrBG')
        ax.set_axis_off()
        cax.set_axis_off()
        ax.set_title("%s distribution of band %s Hz" % (column, band_name))
        plt.savefig(self.output_folder.joinpath('img/spatial_features/%s_%s_distr.png' % (column, band)))
        plt.show()

    def plot_all_features_distr(self, map_file=None):
        """
        Plot all the features distribution and save them in the correspondent folder
        Parameters
        ----------
        map_file : string or path
            Background map. If None, a default one will be added
        """
        for feature in self.features:
            for band_i in np.arange(len(self.band_list)):
                self.plot_distr(column=feature, band=band_i, map_file=map_file)

    def plot_all_features_evo(self, group_by='station_name'):
        """
        Creates the images of the temporal evolution of all the features and saves them in the correspondent folder

        Parameters
        ----------
        group_by: string
            Column in which to separate the plots. A figure will be generated for each group
        """
        i = 0
        for station_name, deployment in self.dataset.groupby(group_by):
            for feature in self.features:
                deployment[feature].plot()
                plt.title('%s %s evolution' % (station_name, feature))
                plt.savefig(
                    self.output_folder.joinpath('img/temporal_features/%s_%s_%s.png' % (i, station_name, feature)))
                plt.show()
                i += 1

    def plot_third_octave_bands_prob(self, group_by='station_name', h=1.0, percentiles=None):
        """
        Create a plot with the probability distribution of the levels of the third octave bands
        Parameters
        ----------
        group_by: string
            Column in which to separate the plots. A figure will be generated for each group
        h: float
            Histogram bin size (in db)
        percentiles: list of floats
            Percentiles to plot (0 to 1). Default is 10, 50 and 90% ([0.1, 0.5, 0.9])
        """
        if percentiles is None:
            percentiles = [0.1, 0.5, 0.9]
        percentiles = np.array(percentiles)
        if self.third_octaves is False:
            raise Exception('This is only possible if third-octave bands have been computed!')
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        bin_edges = np.arange(start=self.dataset['oct3'].min().min(), stop=self.dataset['oct3'].max().max(), step=h)
        fbands = self.dataset['oct3'].columns
        station_i = 0
        for station_name, deployment in self.dataset.groupby((group_by, 'all')):
            sxx = deployment['oct3'].values.T
            spd = np.zeros((sxx.shape[0], bin_edges.size - 1))
            p = np.zeros((sxx.shape[0], percentiles.size))
            for i in np.arange(sxx.shape[0]):
                spd[i, :] = np.histogram(sxx[i, :], bin_edges, density=True)[0]
                cumsum = np.cumsum(spd[i, :])
                for j in np.arange(percentiles.size):
                    p[i, j] = bin_edges[np.argmax(cumsum > percentiles[j] * cumsum[-1])]
            fig = plt.figure()
            im = plt.pcolormesh(fbands, bin_edges[:-1], spd.T, cmap='BuPu', shading='auto')
            # Plot the lines of the percentiles
            plt.plot(fbands, p, label=percentiles)
            plt.xlabel('Frequency [Hz]')
            plt.xscale('log')
            plt.ylabel('$L_{rms}$ [dB]')
            cbar = fig.colorbar(im)
            cbar.set_label('Empirical Probability Density', rotation=90)
            plt.title('1/3-octave bands probability distribution %s' % station_name)
            plt.savefig(self.output_folder.joinpath('img/features_analysis/%s_%s_third_oct_prob.png' %
                                                    (station_i, station_name)))
            plt.show()
            station_i += 1

    def plot_third_octave_bands_avg(self, group_by='station_name'):
        """
        Plot the average third octave bands
        Parameters
        ----------
        group_by: string
            Column in which to separate the plots. A figure will be generated for each group
        """
        if self.third_octaves is False:
            raise Exception('This is only possible if third-octave bands have been computed!')
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        station_i = 0
        for station_name, deployment in self.dataset.groupby((group_by, 'all')):
            deployment['oct3'].mean(axis=0).plot()
            plt.title('1/3-octave bands average %s' % station_name)
            plt.xlabel('Frequency [Hz]')
            plt.xscale('log')
            plt.ylabel('Average Sound Level [dB re 1 $\mu Pa$]')
            plt.savefig(self.output_folder.joinpath('img/features_analysis/%s_%s_third_oct_avg.png' %
                                                    (station_i, station_name)))
            plt.show()
            station_i += 1


class Deployment:
    def __init__(self,
                 hydrophone,
                 station_name,
                 method,
                 hydrophone_depth,
                 data_folder_path,
                 gps_path,
                 utc=False,
                 etn_id=0,
                 include_dirs=True):
        """
            Represents one deployment.
            Parameters
            ----------
            hydrophone : hydrophone object
                Hydrophone object from pyhydrophone with all the characteristics
            station_name : string
                Name of the deployment
            method : string
                Method name of the data acquisition
            hydrophone_depth : float
                Depth under the surface of the hydrophone
            data_folder_path : string or Path
                Folder where the data is
            gps_path : string or Path
                File where the gps data is
        """
        self.hydrophone = hydrophone
        self.station_name = station_name
        self.method = method
        self.hydrophone_depth = hydrophone_depth
        self.data_folder_path = data_folder_path
        self.gps_path = gps_path
        self.utc = not utc
        self.etn_id = etn_id
        self.include_dirs = include_dirs
        self.geoloc = geolocation.SurveyLocation(geofile=self.gps_path)
        self.evo = None

        self.mapdata = mapdata.MapData()
        self.timedata = timedata.TimeData()
        self.seastate = seastatedata.SeaStateData()
        self.humandata = humandata.HumanData()
        self.shipwreck = shipwreck.ShipWreck()

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

    def generate_deployment_data(self, features=None, third_octaves=None, band_list=None, binsize=60.0, nfft=512):
        """
        Generate a dataset in a pandas data frame with the spl values
        Parameters
        ----------
        features: list of str
            All the features to compute
        third_octaves: list or false
            Set to False if no third octaves have to be computed. Pass the band to calculate the third octaves in
            otherwise. If set to None, the broadband [0, fs/2] will be considered. Band is independent of the bands_list
        band_list: list of lists
            List of bands to compute all the features of. If set to None, the broadband [0, fs/2] will be considered.
        binsize: float
            Size of the bins to compute the features from, in seconds
        nfft: int
            Number of samples of the window size (only will be used in case of a frequency-domain operation)

        Returns
        -------
        DataFrame with datetime as index and all the features and octave bands as columns. Other columns such as
        method, hydrophone_depth, instrument_name, instrument_model, instrument_Vpp, instrument_preamp_gain,
        instrument_sensitivity, station_name and etn_id  are also added.
        """
        if self.hydrophone.name == 'B&K':
            # Find and remove calibration tone, calibrate output
            asa = acoustic_survey.ASA(self.hydrophone, self.data_folder_path, binsize=binsize, nfft=nfft,
                                      utc=self.utc, include_dirs=self.include_dirs,
                                      calibration_time='auto', cal_freq=159.2, max_cal_duration=120.0, dc_subtract=True)
            self.hydrophone = asa.hydrophone
        else:
            asa = acoustic_survey.ASA(self.hydrophone, self.data_folder_path, binsize=binsize, nfft=nfft, utc=self.utc,
                                      include_dirs=self.include_dirs, dc_subtract=True)
        if features is None or len(features) == 0:
            if third_octaves is not False:
                evo = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
            else:
                evo = asa.timestamps_df()
        else:
            evo = asa.evolution_multiple(method_list=features, band_list=band_list)
            if third_octaves is not False:
                evo_freq = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
                oct_3bands = evo_freq.loc[:, evo_freq.columns.get_level_values('method') == 'oct3']
                evo = evo.merge(oct_3bands, left_index=True, right_index=True)
        evo[('method', 'all')] = self.method
        evo[('hydrophone_depth', 'all')] = self.hydrophone_depth
        evo[('instrument_name', 'all')] = self.hydrophone.name
        evo[('instrument_model', 'all')] = self.hydrophone.model
        evo[('instrument_Vpp', 'all')] = self.hydrophone.Vpp
        evo[('instrument_preamp_gain', 'all')] = self.hydrophone.preamp_gain
        evo[('instrument_sensitivity', 'all')] = self.hydrophone.sensitivity
        evo[('station_name', 'all')] = self.station_name
        evo[('etn_id', 'all')] = self.etn_id
        evo.index = evo.index.tz_localize('UTC')
        self.evo = evo
        return evo

    def add_spatial_data(self):
        """
        Add the corresponding gps location to each row of the features
        """
        if self.method == 'Moored':
            moored_point = self.geoloc.geotrackpoints.iloc[0].geometry
            self.evo['geom'] = moored_point
            self.evo = geopandas.GeoDataFrame(self.evo, geometry='geom', crs=self.geoloc.geotrackpoints.crs)
            self.evo['geometry'] = self.evo.geometry
        else:
            self.evo = self.geoloc.add_survey_location(self.evo)
        return self.evo

    def add_distance_to(self, lat=None, lon=None, column=''):
        """
        Add a column with the distance to (lat, lon)
        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        column : string
            Name of the column where to save the computed distance
        """
        # Read the gpx (time is in UTC)
        evo_loc = self.geoloc.add_distance_to(df=self.evo,
                                              lat=lat,
                                              lon=lon,
                                              column=column)

        evo_loc = evo_loc.set_index('datetime')
        evo_loc.columns = pd.MultiIndex.from_product([evo_loc.columns, ['all']])
        evo = self.evo.merge(evo_loc, left_index=True, right_index=True)
        self.evo = geopandas.GeoDataFrame(evo, crs=evo_loc.crs)
        return self.evo

    def add_distance_to_coast(self, coastfile):
        """
        Add distance to coast
        Parameters
        ----------
        coastfile : string or Path
            *.shp file with the coastline
        """
        self.evo = self.geoloc.add_distance_to_coast(df=self.evo, coastfile=coastfile, column='coast_dist')
        return self.evo

    def add_seastate(self):
        """
        Add all the physical parameters
        """
        self.evo = self.seastate.get_data(self.evo)
        return self.evo

    def add_time_data(self):
        """
        Add the different data concerning time
        """
        self.evo = self.timedata.get_time_data_df(self.evo)
        return self.evo

    def add_seabottom_data(self):
        """
        Add habitat type to evo
        """
        self.evo = self.mapdata.get_seabottom_data(self.evo, ['Substrate', 'Allcomb'])
        return self.evo

    def add_shipping_data(self):
        """
        Add AIS intensity
        """
        self.evo = self.humandata.get_shipping_intensity_df(self.evo)
        return self.evo

    def add_shipwreck_info(self):
        """
        Add the name of the closest shipwreck as well as the distance
        """
        self.evo = self.shipwreck.get_shipwreck_information_df(self.evo)
        return self.evo
        

    def plot_features_vs_distance_to(self, features, geo_column):
        """
        Generate a dataset in a pandas data frame with the spl values
        Parameters
        ----------
        features : list of strings
            List of all the features to plot
        geo_column : string
            Geometry column (distance) to plot to
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

    def plot_distr_over(self, map_path, map_column=None, deployment_column='geometry'):
        """
        Plot the column "deployment_column" on top of a map data, where map_column is selected as the layout
        Parameters
        ----------
        map_path : string or Path
            Map with the layout
        map_column : string or Path
            Column to choose from the layout
        deployment_column : string
            Column from the evo dataframe to plot. If None geometry is used (all the sampling points)
        """
        map_df = geopandas.read_file(map_path)
        ax = map_df.plot(column=map_column)
        self.evo.to_crs(map_df.crs).plot(column=deployment_column, ax=ax)
        plt.show()

    def plot_temporal(self, feature, save_path=None):
        """
        Plot the temporal evolution of the feature for the entire the deployment
        Parameters
        ----------
        feature : string
            Name of the feature to plot
        save_path : string or Path
            Path where to save the plot. If None, it will not be saved
        """
        self.evo.plot(feature)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def save(self, folder_path):
        """
        Save the deployment evolution in pickle format
        Parameters
        ----------
        folder_path : string or Path
            Folder where to save it. The name will be the date and the station
        """
        date_str = self.evo.index[0].strftime("%d%m%y")
        self.evo.to_pickle(folder_path.joinpath('%s_%s.pkl', (date_str, self.station_name)))
