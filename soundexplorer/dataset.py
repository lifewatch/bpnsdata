import datetime
import pathlib

import contextily as ctx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely
import rasterio
import urllib
from importlib import resources
shapely.speedups.disable()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypam import acoustic_survey, geolocation
from skyfield import almanac, api


class DataSet:
    def __init__(self, summary_path, output_folder, instruments, features, third_octaves=True,
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
        binsize : float
            In seconds, duration of windows to consider
        nfft : int
            Number of samples of window to use for frequency analysis
        """
        self.metadata = pd.read_csv(summary_path)
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
        self.read_dataset()

    def generate_entire_dataset(self, coastfile=None):
        """
        Calculates the acoustic features of every deployment and saves them as a pickle in the deployments folder with
        the name of the station of the deployment.
        Also adds all the deployment data to the self object in the general dataset,
        and the path to each deployment's pickle in the list of deployments
        """
        for index in self.metadata.index:
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.sensitivity = deployment_row['hydrophone_sensitivity']
            d = Deployment(hydrophone=inst,
                           location=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'],
                           datetime_col=deployment_row['datetime_col'],
                           lon_col=deployment_row['lon_col'],
                           lat_col=deployment_row['lat_col'],
                           utc=deployment_row['utc'])
            d.generate_deployment_data(features=self.features, band_list=self.band_list,
                                       third_octaves=self.third_octaves, binsize=self.binsize, nfft=self.nfft)
            d.add_spatial_data()
            if coastfile is not None:
                d.add_distance_to_coast(coastfile=coastfile)
            d.add_seastate()
            d.add_time_data()
            d.add_seabottom_data()
            d.add_shipping_data()
            deployment_path = self.output_folder.joinpath('deployments/%s_%s.pkl' % (index, d.station_name))
            d.evo.to_pickle(deployment_path)
            self.deployments.append(deployment_path)
            self.dataset = self.dataset.append(d.evo)
        self.dataset.to_pickle(self.output_folder.joinpath('dataset.pkl'))

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
        self.dataset = pd.DataFrame()

        for index, deployment_file in enumerate(self.deployments):
            deployment_row = self.metadata.iloc[index]
            inst = self.instruments[deployment_row['instrument']]
            inst.sensitivity = deployment_row['hydrophone_sensitivity']
            d = Deployment(hydrophone=inst,
                           location=deployment_row['location'],
                           method=deployment_row['method'],
                           hydrophone_depth=deployment_row['hydrophone_depth'],
                           data_folder_path=deployment_row['data_folder'],
                           gps_path=deployment_row['gps_path'],
                           datetime_col=deployment_row['datetime_col'],
                           lon_col=deployment_row['lon_col'],
                           lat_col=deployment_row['lat_col'],
                           utc=deployment_row['utc'])
            d.evo = pd.read_pickle(deployment_file)
            d.evo.drop_duplicates(inplace=True)
            if 'geometry' in d.evo.columns:
                d.evo = geopandas.GeoDataFrame(d.evo, geometry='geom', crs='EPSG:4326')
            else:
                d.add_spatial_data()
            # d.add_seastate()
            # d.add_seabottom_data()
            # d.add_time_data()
            # d.add_shipping_data()
            self.dataset = self.dataset.append(d.evo)
            d.evo.to_pickle(deployment_file)
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
                           datetime_col=deployment_row['datetime_col'],
                           lon_col=deployment_row['lon_col'],
                           lat_col=deployment_row['lat_col'])
            metadata.at[index, ['start_datetime', 'end_datetime', 'duration']] = d.get_metadata()
        return metadata

    def plot_distr(self, column, band=None, map_file=None, categorical=False):
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
        categorical : bool
            Set to True if it is a categorical variable
        """
        _, ax = plt.subplots(1, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        if band is None:
            band = 'all'
        if categorical:
            self.dataset.plot(column=(column, band), ax=ax, legend=True, alpha=0.5, categorical=True, cax=cax)
        else:
            self.dataset[(column, band)] = self.dataset[(column, band)].astype(float)
            self.dataset.plot(column=(column, band), ax=ax, legend=True, alpha=0.5, cmap='YlOrRd', categorical=False, cax=cax)
        if map_file is None:
            ctx.add_basemap(ax, crs=self.dataset.crs.to_string(), source=ctx.providers.Stamen.TonerLite,
                            reset_extent=False)
        else:
            ctx.add_basemap(ax, crs=self.dataset.crs.to_string(), source=map_file, reset_extent=False, cmap='BrBG')
        ax.set_axis_off()
        band_name = self.bands_list[band]
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
            for band_i in np.arange(len(self.bands_list)):
                self.plot_distr(column=feature, band=band_i, map_file=map_file)

    def plot_all_features_evo(self, group_by='station_name'):
        """
        Creates the images of the temporal evolution of all the features and saves them in the correspondent folder
        """
        for station_name, deployment in self.dataset.groupby(group_by):
            for feature in self.features:
                deployment[feature].plot()
                plt.title('%s %s evolution' % (station_name, feature))
                plt.savefig(self.output_folder.joinpath('img/temporal_features/%s_%s.png' % (station_name, feature)))
                plt.show()

    def plot_third_octave_bands_prob(self, group_by='station_name', h=1.0, percentiles=[0.1, 0.5, 0.9]):
        """
        Create a plot with the probability distribution of the levels of the third octave bands
        """
        percentiles = np.array(percentiles)
        if not self.third_octaves:
            raise Exception('This is only possible if third-octave bands have been computed!')
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        bin_edges = np.arange(start=self.dataset['oct3'].min().min(), stop=self.dataset['oct3'].max().max(), step=h)
        fbands = self.dataset['oct3'].columns
        for station_name, deployment in self.dataset.groupby((group_by, 'all')):
            sxx = deployment['oct3'].values.T
            spd = np.zeros((sxx.shape[0], bin_edges.size - 1))
            p = np.zeros((sxx.shape[0], percentiles.size))
            for i in np.arange(sxx.shape[0]):
                spd[i, :] = np.histogram(sxx[i, :], bin_edges)[0] / ((bin_edges.size - 1) * h)
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
            plt.savefig(self.output_folder.joinpath('img/features_analysis/%s_third_oct.png' % station_name))
            plt.show()

    def plot_third_octave_bands_avg(self, group_by='station_name'):
        """
        Plot the average
        """
        if not self.third_octaves:
            raise Exception('This is only possible if third-octave bands have been computed!')
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        for station_name, deployment in self.dataset.groupby((group_by, 'all')):
            deployment['oct3'].mean(axis=0).plot()
            plt.title('1/3-octave bands average %s' % station_name)
            plt.xlabel('Frequency [Hz]')
            plt.xscale('log')
            plt.ylabel('Average Sound Level [dB re 1 $\mu Pa$]')
            plt.savefig(self.output_folder.joinpath('img/features_analysis/%s_third_oct.png' % station_name))
            plt.show()


class Deployment:
    def __init__(self,
                 hydrophone,
                 location,
                 method,
                 hydrophone_depth,
                 data_folder_path,
                 gps_path,
                 datetime_col='datetime',
                 lat_col='Latitude',
                 lon_col='Longitude',
                 utc=False):
        """
        Represents one deployment.
        Parameters
        ----------
        hydrophone : hydrophone object
            Hydrophone object from pyhydrophone with all the characteristics
        location : string
            Name of the deployment
        method : string
            Method name of the data acquisition
        hydrophone_depth : float
            Depth under the surface of the hydrophone
        data_folder_path : string or Path
            Folder where the data is
        gps_path : string or Path
            File where the gps data is
        datetime_col : string
            Name of the column where the datetime is in the gps file
        lat_col : string
            Name of the column where the Latitude is
        lon_col : string
            Name of the column where the Longitude is
        """
        self.hydrophone = hydrophone
        self.station_name = location
        self.method = method
        self.hydrophone_depth = hydrophone_depth
        self.data_folder_path = data_folder_path
        self.gps_path = gps_path
        self.utc = not utc
        self.geoloc = geolocation.SurveyLocation(geofile=self.gps_path, datetime_col=datetime_col,
                                                 lat_col=lat_col, lon_col=lon_col)
        self.evo = None

        self.mapdata = MapData()
        self.timedata = TimeData()
        self.seastate = SeaState()
        self.humandata = HumanData()

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

    def generate_deployment_data(self, features=None, third_octaves=True, band_list=None, binsize=60.0, nfft=512):
        """
        Generate a dataset in a pandas data frame with the spl values
        """
        asa = acoustic_survey.ASA(self.hydrophone, self.data_folder_path, binsize=binsize, nfft=nfft, utc=self.utc)
        if features is not None:
            evo = asa.evolution_multiple(method_list=features,
                                         band_list=band_list)
            if third_octaves:
                evo_freq = asa.evolution_freq_dom('third_octaves_levels', binsize=binsize, db=True)
                evo = evo.merge(evo_freq, left_index=True, right_index=True)
        else:
            if third_octaves:
                evo = asa.evolution_freq_dom('third_octaves_levels', binsize=binsize, db=True)
        evo[('method', 'all')] = self.method
        evo[('hydrophone_depth', 'all')] = self.hydrophone_depth
        evo[('instrument_name', 'all')] = self.hydrophone.name
        evo[('instrument_model', 'all')] = self.hydrophone.model
        evo[('instrument_Vpp', 'all')] = self.hydrophone.Vpp
        evo[('instrument_preamp_gain', 'all')] = self.hydrophone.preamp_gain
        evo[('instrument_sensitivity', 'all')] = self.hydrophone.sensitivity
        evo[('station_name', 'all')] = self.station_name
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
                                              column=[column])

        evo_loc = evo_loc.set_index('datetime')
        evo_loc.columns = pd.MultiIndex.from_product([evo_loc.columns, ['all']])
        evo = self.evo.merge(evo_loc, left_index=True, right_index=True)
        self.evo = geopandas.GeoDataFrame(evo, crs=evo_loc.crs)
        return self.evo

    def add_distance_to_coast(self, coastfile, column='coast_dist'):
        """
        Add distance to coast
        Parameters
        ----------
        coastfile : string or Path
            *.shp file with the coastline
        column : string
            Name of the column where to store the distance

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


class TimeData:
    """
    Class to calculate moon phase and moment of the day
    """
    def __init__(self):
        self.ts = api.load.timescale()
        with resources.path('soundexplorer.data', 'de421.bsp') as bsp_file:
            self.eph = api.load_file(bsp_file)

    def get_moon_phase(self, dt, categorical=False):
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
        if categorical:
            moon_phase_at = almanac.moon_phases(self.eph)
            moon_phase = almanac.MOON_PHASES[moon_phase_at(t)]
        else:
            moon_phase = almanac.moon_phase(self.eph, t).radians
        return moon_phase

    def get_day_moment(self, dt, location):
        """
        Return moment of the day (day, night, twilight)
        Parameters
        ----------
        dt : datetime
            Datetime to get the moment of
        location : geometry object

        Returns
        -------
        Moment of the day (string)
        """
        bluffton = api.Topos(latitude_degrees=location.coords[0][0], longitude_degrees=location.coords[0][1])
        utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
        t = self.ts.utc(utc_dt)
        is_dark_twilight_at = almanac.dark_twilight_day(self.eph, bluffton)
        day_moment = is_dark_twilight_at(t).min()
        return almanac.TWILIGHTS[day_moment]

    def get_time_data_df(self, df):
        """
        Add to the dataframe the moon_phase and the day_moment to all the rows
        Parameters
        ----------
        df : DataFrame
            DataFrame with the datetime as index
        Returns
        -------
        The dataframe with the columns added
        """
        df[('moon_phase', 'all')] = None
        df[('day_moment', 'all')] = None
        for t, row in df.iterrows():
            if row.loc[('geometry', '')] is not None:
                df.at[t, ('moon_phase', 'all')] = self.get_moon_phase(t)
                df.at[t, ('day_moment', 'all')] = self.get_day_moment(t, row.loc[('geometry', '')])
            else:
                df.at[t, ('moon_phase', 'all')] = np.nan
                df.at[t, ('day_moment', 'all')] = np.nan
        return df


class SeaState:
    def __init__(self):
        """
        Sea State for BPNS data downloader
        """
        self.seastate = pd.DataFrame()
        self.columns_ph = ['surface_baroclinic_eastward_sea_water_velocity',
                            'surface_baroclinic_northward_sea_water_velocity',
                            'sea_surface_height_above_sea_level',
                            'sea_surface_salinity',
                            'sea_surface_temperature']
        self.columns_wv = ['hs', 'tm_1']
        self.min_lat = 51.0
        self.max_lat = 51.91667
        self.min_lon = 2.083333
        self.max_lon = 4.214285

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
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = df.total_bounds
        start_timestamp = (df.index.min() - datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_timestamp = (df.index.max() + datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        wavestate = self.get_griddap_df(start_timestamp, end_timestamp, 'WAM_ECMWF', self.columns_wv)
        seastate = self.get_griddap_df(start_timestamp, end_timestamp, 'BCZ_HydroState_V1', self.columns_ph)
        for col in self.columns_ph + self.columns_wv:
            df[(col, 'all')] = None

        time_indexes = pd.DataFrame(index=seastate.time.unique())
        moored = len(df.geometry.unique()) == 1
        closest_point_s = None
        closest_point_w = None
        for t, row in df.iterrows():
            if row.loc[('geometry', '')] is not None:
                selected_time_idx = time_indexes.index.get_loc(t, method='nearest')
                selected_time = time_indexes.iloc[selected_time_idx].name
                seastate_t = seastate[seastate.time == selected_time]
                if moored:
                    if closest_point_s is None:
                        closest_point_s = shapely.ops.nearest_points(seastate_t.geometry.unary_union,
                                                                     row.loc[('geometry', '')])[0]
                else:
                    closest_point_s = shapely.ops.nearest_points(seastate_t.geometry.unary_union,
                                                                 row.loc[('geometry', '')])[0]
                closest_row = seastate_t[(seastate_t.latitude == closest_point_s.coords.xy[1]) &
                                         (seastate_t.longitude == closest_point_s.coords.xy[0])]
                df.loc[t, (self.columns_ph, 'all')] = closest_row[self.columns_ph].values[0]
                wavestate_t = wavestate[wavestate.time == selected_time]
                if moored:
                    if closest_point_w is None:
                        closest_point_w = shapely.ops.nearest_points(wavestate_t.geometry.unary_union,
                                                                     row.loc[('geometry', '')])[0]
                else:
                    closest_point_w = shapely.ops.nearest_points(wavestate_t.geometry.unary_union,
                                                                 row.loc[('geometry', '')])[0]
                closest_row = wavestate_t[(wavestate_t.latitude == closest_point_w.coords.xy[1]) &
                                          (wavestate_t.longitude == closest_point_w.coords.xy[0])]
                df.loc[t, (self.columns_wv, 'all')] = closest_row[self.columns_wv].values[0]
            else:
                df.loc[t, (self.columns_wv, 'all')] = np.nan
        df['surface_baroclinic_sea_water_velocity'] = np.sqrt((df[['surface_baroclinic_eastward_sea_water_velocity',
                                                                       'surface_baroclinic_northward_sea_water_velocity']] ** 2).sum(axis=1))

        return df

    def get_griddap_df(self, start_timestamp, end_timestamp, table_name, columns):
        """
        Returns a DataFrame with the data from the griddap service between the specified dates, and the specified
        columns from the specified table
        Parameters
        ----------
        start_timestamp : datetime
            Start time to downlowad the data from
        end_timestamp : datetime
            End time to download the data from
        table_name : string
            Name of the table
        columns : list of strings
            List of all the columns to store

        Returns
        -------
        DataFrame with the downloaded table from griddap
        """
        query = 'https://erddap.naturalsciences.be/erddap/griddap/%s.json?' % table_name
        for col in columns:
            query = query + '%s[(%s):1:(%s)][(%s):1:(%s)][(%s):1:(%s)],' % (col, start_timestamp, end_timestamp,
                                                                            self.min_lat, self.max_lat,
                                                                            self.min_lon, self.max_lon)
        response = requests.get(query[:-1])
        griddap = pd.DataFrame(columns=response.json()['table']['columnNames'], data=response.json()['table']['rows'])
        griddap.dropna(inplace=True)
        griddap.time = pd.to_datetime(griddap.time)
        griddap = geopandas.GeoDataFrame(griddap, geometry=geopandas.points_from_xy(griddap.longitude, griddap.latitude),
                                         crs="EPSG:4326")
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
        square = shapely.geometry.box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)
        return df.geometry.intersects(square).all()


class MapData:
    """
    Class to het the spatial data. Default is habitat type
    """
    def __init__(self, map_path=None):
        """

        Parameters
        ----------
        map_path : string or Path
            Path where the data is
        """
        if map_path is None:
            map_path = pathlib.Path('//fs/shared/datac/Geo/Layers/Belgium/habitatsandbiotopes/'
                                    'broadscalehabitatmap/seabedhabitat_BE.shp')
        else:
            if not isinstance(map_path, pathlib.Path):
                map_path = pathlib.Path(map_path)
        self.map_path = map_path
        self.map = geopandas.read_file(self.map_path)

    def get_location_map_data(self, columns, location, crs):
        """
        Get the features of the columns at a certain location
        Parameters
        ----------
        columns : list of strings
            Columns to get the features from
        location : geometry object
            Location
        crs : string or object
            CRS projection

        Returns
        -------
        List of features in that particular location
        """
        project = pyproj.Transformer.from_crs(crs.geodetic_crs, self.map.crs, always_xy=True).transform
        location_crs = shapely.ops.transform(project, location)
        mask = self.map.contains(location_crs)
        idx = np.where(mask)[0][0]
        features = self.map.loc[idx, columns]
        return features.values

    def get_seabottom_data(self, df, columns):
        """
        Get the features and the bathymetry together of all the DataFrame
        Parameters
        ----------
        df : DataFrame
            Where to add the seabottom data
        columns : list of strings
            Columns from the map to add
        Returns
        -------
        DataFrame with features and bathymetry added
        """
        for column in columns:
            df[(column, 'all')] = None
        diff_points = df.geometry.unique()
        for point in diff_points:
            if point is not None:
                if len(diff_points) == 1:
                    idxes = df.index
                else:
                    idxes = df[df[('geometry', '')] == point].index
                df.loc[idxes, (columns, 'all')] = self.get_location_map_data(columns, point, crs=df.crs)
                df.loc[idxes, ('bathymetry', 'all')] = self.get_bathymetry(point.coords)
            else:
                idxes = df[df[('geometry', '')] == None].index
                df.loc[idxes, (columns, 'all')] = np.nan
                df.loc[idxes, ('bathymetry', 'all')] = np.nan
        return df

    @staticmethod
    def get_bathymetry(location):
        """
        Return the bathymetry of a certain point from EMODnet
        Parameters
        ----------
        location : geometry object
            Location

        Returns
        -------
        Bathymetry as a positive number
        """
        point_str = str(location)
        response = requests.get("https://rest.emodnet-bathymetry.eu/depth_sample?geom=%s" % point_str)
        if response.status_code == 200:
            depth = response.json()['avg']
        else:
            depth = 0.0
        return depth


class HumanData:
    def get_shipping_intensity_df(self, df):
        """
        Will return the intensity of the year and month of each sample
        """
        bbox = df.total_bounds
        # Make sure the bbox is big enough to download some data
        if abs(bbox[2] - bbox[0]) <= 0.00833333 * 2:
            bbox[0] -= 2 * 0.00833333
            bbox[2] += 2 * 0.00833333
        if abs(bbox[3] - bbox[1]) <= 0.00833333 * 2:
            bbox[1] -= 2 * 0.00833333
            bbox[3] += 2 * 0.00833333
        bbox = ",".join(bbox.astype(str))
        years = df.index.year.unique()
        months = df.index.month.unique()
        df[('route_dens', 'all')] = None
        for year in years:
            for month in months:
                request = 'https://ows.emodnet-humanactivities.eu/wcs?service=wcs&version=1.0.0&request=getcoverage&' \
                          'coverage=emodnet:%s_%02d_rd_All&crs=EPSG:4326&BBOX=%s&' \
                          'format=image/tiff&interpolation=nearest&resx=0.00833333&resy=0.00833333' % (year, month, bbox)
                response = requests.get(request)
                if response.status_code != 200:
                    print('No shipping density layer found for %s-%s. Rows values set to None' % (year, month))
                else:
                    try:
                        tif_file = urllib.request.urlretrieve(request)
                        tif_raster = rasterio.open(tif_file[0])
                        for idx, row in df.iterrows():
                            if row['geometry', ''] is not None:
                                if idx.month == month and idx.year == year:
                                    x = row['geometry', ''].xy[0][0]
                                    y = row['geometry', ''].xy[1][0]
                                    row, col = tif_raster.index(x, y)
                                    df.at[idx, ('route_dens', 'all')] = tif_raster.read(1)[row, col]
                    except:
                        print('Year %s and month %s was not downloaded' % (year, month))
                        df.at[idx, ('route_dens', 'all')] = np.nan
        return df
