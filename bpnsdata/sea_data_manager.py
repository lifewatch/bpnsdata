import bpnsdata
from bpnsdata import geolocation
import shapely

shapely.speedups.disable()

import warnings
warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)


class SeaDataManager:
    def __init__(self, env_vars):
        """
        Start a Sea Data Manager with a list of all the environment variables that want to be added
        Parameters
        ----------
        env_vars : list
            List with all the environmental variables. It can be any class present in BPNS, with the name in lowercase
            and the words separated with an underscore. The 'Data' at the end should be ignored.
            i.e: for the SeabedHabitatData class, the value to pass should be seabed_habitat
        """
        self.env_vars = {}
        self.survey_location = None
        for env in env_vars:
            env_class = ''.join([i.capitalize() for i in env.split('_')]) + 'Data'
            self.env_vars[env] = getattr(bpnsdata, env_class)()

    def __call__(self, df, datetime_column='datetime', verbose=False, **kwargs):
        """
        Calls the function of all the env_vars of the class to add the parameters to the df.
        Parameters
        ----------
        df : GeoDataFrame
            Geopandas DataFrame with one column with geometry and the datetime as index. Some classes do not need
            geometry or datetime
        datetime_column: str
            Column where the datetime information is stored. Default to 'datetime'
        verbose: bool
            Set to True to get information about what is running
        kwargs : dict
            key has to be the name of one of the env_vars and value is a dictionary with key, value as the parameters
            that can be passed to the __call__ function of that class.

        Returns
        -------
        The updated df
        """
        if 'geometry' not in df.columns:
            raise UserWarning('This dataframe does not have a geometry. Please add the geometry. It can be done using '
                              'the function add_geodata')
        for env_name, env_class in self.env_vars.items():
            if verbose:
                print('Adding %s...' % env_name)
            if env_name in kwargs.keys():
                df = env_class(df, datetime_column=datetime_column, **kwargs[env_name])
            else:
                df = env_class(df, datetime_column=datetime_column)
        return df

    def add_geodata(self, df, geofile, datetime_column='datetime', time_tolerance='1min', other_cols=None):
        """
        Add the geo information to the df from a geofile with the time and space information about the df.
        See SurveyLocation for more information

        Parameters
        ----------
        df : DataFrame
            Pandas dataframe with datetime as index
        geofile : str or Path
            Where the geo data is stored. can be a csv, gpx, pickle or sqlite3
        datetime_column: str
            Column where the datetime information is stored. Default to 'datetime'
        time_tolerance : str
            Time tolerance with the geolocation. i.e. '10min' or '1s'
        other_cols : list or None
            list of the columns from the geofile to keep a part from geometry

        Returns
        -------
        The same dataframe converted to a geodataframe with the updated geometry. The index kept are the original ones
        from the DataFrame, not the ones from the geo file.
        """
        self.survey_location = geolocation.SurveyLocation(geofile)
        geodf = self.survey_location.add_location(df, datetime_column=datetime_column,
                                                  time_tolerance=time_tolerance, other_cols=other_cols)
        return geodf
