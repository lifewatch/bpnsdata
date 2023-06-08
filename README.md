# BPNSdata 

bpnsdata is a package to add environmental data to a geopandas DataFrame. There is no support for multiindex columns, 
so one level has to be selected or dropped out before using it.
Right now only Belgian Part of the North Sea data is available for all the classes. However, some classes are not 
restricted to the bpns and can be used to add environmental data to other parts of the world. 

## Install 
```bash
pip install bpnsdata
```
Installing the requirements can be a bit tricky, so it might fail during installation if you are working on Windows.
If that is the case,we recommend to install FIRST the following packages (in this order) by downloading the wheels of: 
* GDAL
* rasterio
* Fiona

You can follow this tutorial if you're not familiar with wheels and/or pip: 
https://geoffboeing.com/2014/09/using-geopandas-windows/


## Environmental data
Environmental data can be added by specifying it in the env_vars variable when calling the main class SeaDataManager.
To do so, it is necessary to have gps information, which can be stored in a .gpx file, in the "waypoints" or 
"track_points" layer. It can also be loaded as a csv or a shp file.
Then the algorithm finds the point which is closest in time for row of the dataframe.
The available data sources are: 
* csv: Static geo data in a csv file. Files to be provided by the user
* time: Information about the moment of the day and the moon
    * moon cycle
    * day moment
* emodnet: wcs data from EMODnet
    * shipping density
    * bathymetry
* raster: raster data (tiff images)
    * seabed habitats
    * habitat suitability
* griddap: RBINS data from the erddap server
    * sea surface 
    * wave information
* wrakken_bank: shipwreck information 
* meetnet_vlaamse_banken: read weather data from the buoys of the meetnet vlaamse banken
* ais: AIS data from the AIS hub from VLIZ. Only access when connected to the VPN of VLIZ for the moment.

For easier running of the classes, there is a main class called SeaDataManager, which allows to run all the 
desired environmental variables in one line of code.

### EMODnet
Entry point to download map data from EMODnet using WCS. Coverage to be checked in EMODnet, but larger than BPNS. 
The implemented classes so far are: 

##### Shipping (class ShippingData)
Shipping activity from https://www.emodnet-humanactivities.eu/
Adds the route density or the shipping intensity from the month of the deployment to the dataset, considering the 
location, the year and the month. 
It adds the columns: 
* route_density
* ship_density
(depending on the layer type selected)

##### Bathymetry (class BathymetryData)
Adds the mean bathymetry (https://www.emodnet-bathymetry.eu/) layer considering location (no time considered)
The output column is:
* bathymetry

### Raster Data 
Raster Data represents geographical data. Only BPNS available 
The two outputs are:

##### Seabed habitats (class SeabedHabitatsData)
Adds the sea habitat (https://www.emodnet-seabedhabitats.eu/).
The output columns are: 
* seabed_habitat
* substrate

##### Benthic habitats (class BenthicHabitatsData)
Habitat suitability map from the publication ([1]V. Van Lancker, G. Moerkerke, I. Du Four, E. Verfaillie, M. Rabaut, 
and S. Degraer, “Fine-scale Geomorphological Mapping of Sandbank Environments for the Prediction of Macrobenthic 
Occurences, Belgian Part of the North Sea,” Seafloor Geomorphology as Benthic Habitat, 
pp. 251–260, 2012, doi: 10.1016/B978-0-12-385140-6.00014-1.). 
The closest point from the maps is added to the each point of the dataset. The output column is: 
* benthic_habitat

### ERDDAP RBINS Data
Sea State Data from RBINS (https://erddap.naturalsciences.be/erddap/index.html). 
Coverage to be checked in the RBINS erddap website, but restricted to North Sea. 
In this version only the tables BCZ_HydroState_V1 and WAM_ECMWF are implemented.

##### Sea Surface (class SeaSurfaceData)
The data is added from the table: BCZ_HydroState_V1.
* surface_baroclinic_eastward_sea_water_velocity
* surface_baroclinic_northward_sea_water_velocity
* sea_surface_height_above_sea_level
* sea_surface_salinity
* sea_surface_temperature
* surface_baroclinic_sea_water_velocity

##### Sea Bottom (class SeaBottomData)
The data is added from the table: BCZ_HydroState_V1.
* bottom_baroclinic_eastward_sea_water_velocity
* bottom_baroclinic_northward_sea_water_velocity
* bottom_upward_sea_water_velocity

##### Wave Data (class WaveData)
The data is added from the table: WAM_ECMWF
Output columns:
* hs: wave height in cm
* tm_1: wave period
    

### Time Data (class TimeData)
Data Related to time series. It adds the time of the day (day, night, twilight dawn...) and the moon phase. 
The calculation is done using skyfield (https://rhodesmill.org/skyfield/). 
Coverage in all the world. 
The output columns are: 
* moment_day (twilight, dawn, day, night)
* moon_phase (in radians)

### Csv Data (class CSVData)
Static data that is stored in a csv, with a lat and a lon columns (names to be given).
It returns the closest point of all the csv, the distance to it, the coordinates and also other columns selected by the
user with the specified suffix.

### Wrakken Bank (class WrakkenBankData)
Will add information about the closest shipwreck. The data is extracted from https://wrakkendatabank.afdelingkust.be/.
Following information will be added:
* shipwreck_distance: Distance to closest shipwreck
* shipwreck_lat 
* shipwreck_lon
* shipwreck_name

### Meetnet Vlaamse Banken
Read the available weather forecast at the closest buoy from https://api.meetnetvlaamsebanken.be/V2-help/.
Attention! To be able to use this feature you need to have a user registered at Meet Net Vlaamse Banken. You can do
it for free from their webpage. Then you need the username and the password. You can pass it directly to the created 
objects, but if you want to use them in the SeaDataManager you will have to add the username and the password as 
environmental variables (username_bank and password_bank).
So far, rainfall (NSI) and average wind speed at 10 m (WVC) are implemented. 
It adds to the DataFrame a column with the value of the data, the id of the specified buoy and the distance to the buoy.
The id of the buoy is represented by the sum of the location id + the data id. i.e., in the buoy OMP, the id for 
precipitation is OMP+NSI=OMPNSI

#### Rain (class RainData)
Rainfall in NSI at the closest buoy

#### Wind (class WindData)
Average wind speed at 10 m from the surface, at the closest buoy

### AIS (class AISData)
AIS data from the AIS hub from VLIZ. Only access when connected to the VPN of VLIZ for the moment.
Adds the columns: 
* ais_total_seconds: total cumulative seconds when a ship was there 
* ais_n_ships: total number of ships
* ais_total_seconds_distance_weighted: total cumulative number of seconds, weighted according to the distance of
each ship


## Usage 
Possible ways of loading the data. By default, all the classes read the column 'datetime' as the column from the 
GeopandasDataFrame where the time information is stored, but the user can select another column by specifying the 
datetime_column argument. 

```python 
import bpnsdata 
import pandas as pd 
import numpy as np

# When the desired df is already on a gpx or a csv with coordinates (in this case, imagine the gpx itself 
# contains the rows to analyze
geofile = 'data/VG.gpx'
geodf = bpnsdata.SurveyLocation(geofile).geotrackpoints

# Could also be done directly using geopandas: 
geodf = geopandas.read_file(geofile)

# Could be that we have a df (here a random one) and we want to add a geolocation to it
# Create a random dataframe to work with
time_index = pd.date_range(start='2020-10-12 11:35', end='2020-10-12 12:00', freq='m', tz='UTC')
random_data = np.random.randint(5, 30, size=10)
df = pd.DataFrame(random_data)
df['datetime'] = time_index
```

All the classes can be used separately, by calling each class in its own. 
First declare the class with the desired parameters, then call the object with the df as an argument. 

```python 
import bpnsdata

# For example, for shipping: 
shipping = bpnsdata.ShippingData(layer_name='routedensity', boat_type='all')
df_env = shipping(geodf)
```


### Use of the SeaDataManager

The SeaDataManager can be used when multiple env parameters have to be added. 
Then the user needs to list which ones need to be added. These names are the names of the available classes but with 
all small letters and an underscore separating the words, and removing the Data at the end.

For example: 
* TimeData -> time
* HabitatSuitabilityData -> habitat_suitability 
* SeaSurfaceData -> sea_surface

```python 
import bpnsdata

# Define the seadatamanager
env_vars = ['shipping', 'time', 'wrakken_bank', 'habitat_suitability', 'bathymetry'
            'seabed_habitat', 'sea_surface', 'sea_bottom', 'sea_wave', 'rain', 'wind', 'ais', 'sea_wave_north_sea']
manager = bpnsdata.SeaDataManager(env_vars)

# If the data is without geometry, then:
geodf = manager.add_geodata(df, gpx_file)

# Once the data has geometry:
df_env = manager(geodf)
```
If specific parameters want to be passed, then you can pass a dictionary when calling the SeaDataManager, where the 
key has to be the name of one of the env_vars and value is a dictionary with key, value as the parameters that can 
be passed to the __call__ function of that class.
```python 
import bpnsdata

# Define the seadatamanager
env_vars = ['shipping', 'time', 'wrakken_bank', 'habitat_suitability', 'bathymetry'
            'seabed_habitat', 'sea_surface', 'sea_bottom', sea_wave', 'rain', 'wind', 'ais']
manager = bpnsdata.SeaDataManager(env_vars, {'ais': {'buffer': 20000}})

# Call the seadatamanager 
env_df = manager(geodf)
```