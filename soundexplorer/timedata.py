import datetime
from importlib import resources

import numpy as np
from skyfield import almanac, api


class TimeData:
    """
    Class to calculate moon phase and moment of the day
    """

    def __init__(self):
        self.ts = api.load.timescale()
        try:
            with resources.path('soundexplorer.data', 'de421.bsp') as bsp_file:
                self.eph = api.load_file(bsp_file)
        except FileNotFoundError:
            self.eph = api.load('de421.bsp')

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
            if row[('geometry', '')] is not None:
                df.loc[t, ('moon_phase', 'all')] = self.get_moon_phase(t)
                df.loc[t, ('day_moment', 'all')] = self.get_day_moment(t, row[('geometry', '')])
            else:
                df.loc[t, ('moon_phase', 'all')] = np.nan
                df.loc[t, ('day_moment', 'all')] = np.nan
        return df
