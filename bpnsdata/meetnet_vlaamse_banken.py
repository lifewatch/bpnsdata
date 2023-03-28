import os
import geopandas
import pandas as pd
import requests
import shapely
from tqdm import tqdm
import numpy as np

TIME_TOLERANCE = '30min'


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class MeetNetVlaamseBanken:
    def __init__(self, user=None, password=None, data_field=None):
        """
        Username and password to access the data from https://api.meetnetvlaamsebanken.be/.
        DataField has to be one of the available acronyms:

        GH1 10% highest waves
        GHA Wave height
        GTZ Average wave period
        H01 1% wave height
        HLF Height waves with period > 10 s
        HM0 Significant wave height
        HUM Relative humidity
        LDR Air pressure
        NSI Precipitation
        RHF High frequent wave direction
        RLF Low frequent wave direction
        SR3 Average current direction
        SV3 Average current velocity
        TGR Ground temperature
        TLU Air temperature
        TNB Wet bulb Temperature
        TZW Sea water temperature
        VHL 'en.$Parameter.Name.VHL'
        VHM 'en.$Parameter.Name.VHM'
        WC1 Max 1 sec gust of wind at 10 m
        WC3 Max 3 sec wind speed at 10 m
        WG3 Max 3 sec gust of wind (at sensor height).
        WRS Average wind direction
        WS5 Tide TAW
        WVC Average wind speed at 10 m
        WVS Average wind speed (at sensor height)
        ZON Solar radiation

        Parameters
        ----------
        user : string
            Username
        password : string
            password
        data_field : string
            One of the acronyms
        """
        if user is None:
            try:
                user = os.environ["username_banken"]
            except KeyError:
                raise ValueError('To access the data from meetnet vlaamsebanken you need to register. '
                                 'Please add your username and password as environment variables as username_banken'
                                 'and password_banken or pass them as arguments when creating a MeetNetVlaamseBanken '
                                 'object')
        if password is None:
            password = os.environ["password_banken"]
        self.user = user
        self.password = password
        self.data_field = data_field
        self.url = 'https://api.meetnetvlaamsebanken.be'
        self.auth = self.login()

    def login(self):
        """
        Get the access token.

        Returns
        -------
        BearerAuth object with the correct access token, which can be used for authentication
        """
        url = self.url + '/Token'
        response = requests.post(url, {'username': self.user, 'password': self.password, 'grant_type': "password"})
        if response.status_code != 200:
            raise Exception('This username and password are not valid to access the meetnet vlaamse banken data!'
                            'Be sure to register yourself in https://api.meetnetvlaamsebanken.be/V2-help/')
        return BearerAuth(response.json()['access_token'])

    def __call__(self, df):
        """
        Add the values of the data_field to the df. It will also add field_id (for location) and field_distance (in m)
        Parameters
        ----------
        df : GeoDataFrame
            Pandas dataframe with datetime as index and a geometry column

        Returns
        -------
        the same GeoDataFrame updated
        """
        df = df.assign(**{self.data_field: np.nan})
        df = df.assign(**{self.data_field + '_id': np.nan})
        try:
            catalogue = self.get_catalog()
            start_time = df.index.min()
            end_time = df.index.max()
            nearest_locations = geopandas.sjoin_nearest(df[['geometry']].to_crs(epsg=3395),
                                                        catalogue.to_crs(epsg=3395),
                                                        how='inner', distance_col=self.data_field + '_distance')
            ids = nearest_locations.id.unique()
            total_values = self.get_data(ids, start_time, end_time)
            for location_id, df_id in nearest_locations.groupby('id'):
                df_values = total_values[total_values.id == location_id]
                df_w_values = pd.merge_asof(df_id, df_values, left_index=True, right_on='datetime',
                                            tolerance=pd.Timedelta(TIME_TOLERANCE), direction='nearest')
                df.loc[df_w_values.index, [self.data_field,
                                           self.data_field + '_distance',
                                           self.data_field + '_id']] = df_w_values[['value',
                                                                                    self.data_field + '_distance',
                                                                                    'id_x']].values
        except requests.exceptions.ConnectionError:
            print('Connection with the Meetnet Vlaamse Blanken could not be done. Setting data to nan...')

        return df

    def get_catalog(self):
        """
        Download all the locations from the specified data parameters (self.data_field)
        
        Returns
        -------
        DataFrame with the locations available for the specific type of data
        """
        url = self.url + '/V2/catalog'
        response = requests.get(url, auth=self.auth)
        buoys = geopandas.GeoDataFrame(columns=['id_location', 'name', 'type', 'geometry'], crs='epsg:4326')
        available_data = pd.DataFrame(columns=['id', 'data_type', 'id_location'])
        if response.status_code == 200:
            catalogue = response.json()
            for data_i in catalogue['AvailableData']:
                if data_i['Parameter'] == self.data_field:
                    available_data.loc[len(available_data)] = data_i['ID'], data_i['Parameter'], data_i['Location']
            unique_locations = available_data.id_location.unique()
            for b in catalogue['Locations']:
                if b['ID'] in unique_locations:
                    idx = len(buoys)
                    buoys.loc[idx, 'geometry'] = shapely.wkt.loads(b['PositionWKT'])
                    buoys.loc[idx, ['id_location', 'name', 'type']] = b['ID'], b['Name'][1]['Message'],\
                                                                      b['Description'][1]['Message']

        return buoys.merge(available_data, on='id_location')

    def get_data(self, ids, start_time, end_time):
        """
        Download the data from these ids in the specified time period

        Parameters
        ----------
        ids: list of strings
            list of all the locations (location + data field is the id)
        start_time : datetime
            Start of the period to download
        end_time : datetime
            End of the period to download

        Returns
        -------
        Dataframe with the columns id, datetime and value, where value is the value of the field at that timestamp
        """
        url = self.url + '/V2/getData'
        response = requests.post(url, {'IDs': ids, 'StartTime': start_time, 'EndTime': end_time}, auth=self.auth)
        data = pd.DataFrame(columns=['id', 'datetime', 'value'])
        if response.status_code == 200:
            values = response.json()['Values']
            if values is not None:
                for id_val in values:
                    for v in tqdm(id_val['Values']):
                        data.loc[len(data)] = id_val['ID'], v['Timestamp'], v['Value']
                data['datetime'] = pd.to_datetime(data['datetime'])
                if data['datetime'].dt.tz is None:
                    data['datetime'] = data['datetime'].dt.tz_localize('UTC')
        return data


class RainData(MeetNetVlaamseBanken):
    def __init__(self, user=None, password=None):
        super().__init__(user, password, data_field='NSI')


class WindData(MeetNetVlaamseBanken):
    def __init__(self, user=None, password=None):
        super().__init__(user, password, data_field='WVC')
