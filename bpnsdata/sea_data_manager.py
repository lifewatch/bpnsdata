from bpnsdata import mapdata, timedata, seastatedata, humandata, shipwreck, geolocation


class SeaDataManger:
    def __init__(self, region='Belgian', map_path=None, borders_name='Belgian', benthic_path=None):
        self.mapdata = mapdata.MapData()
        self.timedata = timedata.TimeData()
        self.seastate = seastatedata.SeaStateData()
        self.humandata = humandata.HumanData()
        self.shipwreck = shipwreck.ShipWreck()

    def __call__(self, df, env_vars):
        if 'geometry' not in df.columns:
            raise UserWarning('This dataframe does not have a geometry. Please add the geometry. It can be done using '
                              'the function add_geodata')
        for env in env_vars:
            df = self.__dict__[env](df)

        return df

    def add_geodata(self, df, geofile):
        survey_location = geolocation.SurveyLocation(geofile)
        geodf = survey_location.add_location(df)
        return geodf
