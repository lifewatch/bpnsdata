import datetime
import sys

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

import numpy as np
from skyfield import almanac, api


class TimeData:
    """
    Class to calculate moon phase and moment of the day
    """
    def __init__(self):
        self.ts = api.load.timescale()
        self.bsp_file = importlib_resources.files('bpnsdata') / 'data' / 'de421.bsp'
        if not self.bsp_file.exists():
            print('Downloading the de421.bsp file...')
            load = api.Loader(self.bsp_file.parent)
            load.download('de421.bsp')

        self.eph = None

    def __call__(self, df, datetime_column='datetime'):
        """
        Add to the df the day_moment and the moon_phase columns

        Parameters
        ----------
        df: pd.DataFrame
            The DataFrame to add the data to
        datetime_column: str
            The name of the column containing the date and time information (in datetime format)
        """
        self.eph = api.load_file(self.bsp_file)
        df = self.get_time_data_df(df, datetime_column)
        self.eph.close()
        return df

    def get_moon_phase(self, dt, categorical=False):
        """
        Return the moon phase of a certain date in radians or in MOON_PHASES = ['New Moon','First Quarter','Full Moon',
        'Last Quarter'] if categorical is true

        Parameters
        ----------
        dt: datetime object
            Datetime on which to calculate the moon phase
        categorical : boolean
            Set to True to get the moon phase name as a string
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
        bluffton = api.Topos(latitude_degrees=location.coords[0][1], longitude_degrees=location.coords[0][0])
        utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
        t = self.ts.utc(utc_dt)
        is_dark_twilight_at = almanac.dark_twilight_day(self.eph, bluffton)
        day_moment = is_dark_twilight_at(t).min()
        return almanac.TWILIGHTS[day_moment]

    def get_twilight_events(self, min_dt, max_dt, location):
        """
        Return moment of the day (day, night, twilight)
        Parameters
        ----------
        min_dt : datetime
            Datetime when to start looking for events
        max_dt : datetime
            Datetime when to stop looking for events
        location : geometry object

        Returns
        -------
        Moment of the day (string)
        """
        bluffton = api.Topos(latitude_degrees=location.coords[0][1], longitude_degrees=location.coords[0][0])
        min_utc_dt = min_dt.replace(tzinfo=datetime.timezone.utc)
        max_utc_dt = max_dt.replace(tzinfo=datetime.timezone.utc)
        t_min = self.ts.utc(min_utc_dt)
        t_max = self.ts.utc(max_utc_dt)
        f = almanac.dark_twilight_day(self.eph, api.Topos(latitude_degrees=location.coords[0][1],
                                                          longitude_degrees=location.coords[0][0]))
        times, events = almanac.find_discrete(t_min, t_max, f)
        first_event = f(t_min).item()

        return first_event, times, events

    def get_time_data_df(self, df, datetime_column):
        """
        Add to the dataframe the moon_phase and the day_moment to all the rows
        Parameters
        ----------
        df : DataFrame
            DataFrame with the datetime as index
        datetime_column: str
            The name of the column containing the date and time information (in datetime format)
        Returns
        -------
        The dataframe with the columns added
        """
        df['moon_phase'] = np.nan
        df['day_moment'] = np.nan
        df_4326 = df.to_crs(epsg=4326)
        for t, time_df in df_4326.groupby(datetime_column):
            df.loc[time_df.index, 'moon_phase'] = self.get_moon_phase(t)
        for _, geometry_df in df_4326.loc[~df_4326.geometry.is_empty].groupby(time_df['geometry'].to_wkt()):
            if geometry_df.geometry is not None:
                # If only one date
                if geometry_df.datetime.min() == geometry_df.datetime.max():
                    df.loc[geometry_df.index, 'day_moment'] = self.get_day_moment(geometry_df.datetime.max(),
                                                                                  geometry_df.geometry.values[0])
                # If multiple dates
                else:
                    previous_e, times, events = self.get_twilight_events(geometry_df[datetime_column].min(),
                                                                         geometry_df[datetime_column].max(),
                                                                         geometry_df.geometry.values[0])
                    previous_time = geometry_df[datetime_column].min()
                    for t, e in zip(times, events):
                        geometry_df.loc[(geometry_df[datetime_column] > previous_time) &
                                        (geometry_df[datetime_column] < t)] = almanac.TWILIGHTS[previous_e]
                        previous_e = e
                    df.loc[geometry_df.index, 'day_moment'] = self.get_day_moment(t, geometry_df.geometry.values[0])
        return df
