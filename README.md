# BPNSdata 

bpnsdata is a package to add environmental data to a geopandas DataFrame. 
Right now only Belgian Part of the North Sea data is available for all the classes. 

## Install 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install 
the dependencies 

```bash
pip install -r requirements.txt 
```

Build the project

```bash
python setup.py install
```

## Environmental data
Environmental data can be added by specifying it in the env_vars variable when calling the main class SeaDataManager.
To do so, it is necessary to have gps information, which can be stored in a .gpx file, in the "waypoints" or 
"track_points" layer. It can also be loaded as a csv or a shp file.
Then the algorithm finds the point which is closest in time for row of the dataframe.
The available data sources are: 
* csv: Static geo data in a csv file
** shipwrecks
* time: Information about the moment of the day and the moon
** moon cycle
** day moment
* emodnet: wcs data from EMODnet
** shipping density
** bathymetry
* raster: raster data (tiff images)
** seabed habitats
** habitat suitability
* griddap: RBINS data from the erddap server
** sea surface 
** wave information

For easier running of the classes, there is a main class called SeaDataManager, which allows to run all the 
desired environmental variables in one line of code: 

```python
import bpnsdata 

env_vars = ['shipping', 'shipwreck', 'time', 'shipwreck', 'habitat_suitability',
            'seabed_habitat', 'sea_surface', 'sea_wave']
manager = bpnsdata.SeaDataManger(env_vars)
manager(geodf)
```

### EMODnet
Entry point to download map data from EMODnet using WCS. The implemented classes so far are: 

#### Shipping
Shipping activity from https://www.emodnet-humanactivities.eu/
Adds the route density or the shipping intensity from the month of the deployment to the dataset, considering the 
location, the year and the month. 

#### Bathymetry
Adds the mean bathymetry (https://www.emodnet-bathymetry.eu/) layer considering location (no time considered)

### Raster Data 
Raster Data represents geographical data. The two outputs are

#### Seabed habitats
Adds the sea habitat (https://www.emodnet-seabedhabitats.eu/)

#### Benthic habitats 
Habitat suitability map from the publication ([1]V. Van Lancker, G. Moerkerke, I. Du Four, E. Verfaillie, M. Rabaut, 
and S. Degraer, “Fine-scale Geomorphological Mapping of Sandbank Environments for the Prediction of Macrobenthic 
Occurences, Belgian Part of the North Sea,” Seafloor Geomorphology as Benthic Habitat, 
pp. 251–260, 2012, doi: 10.1016/B978-0-12-385140-6.00014-1.). 
The closest point from the maps is added to the each point of the dataset 


### ERDDAP RBINS Data
Sea State Data from RBINS (https://erddap.naturalsciences.be/erddap/index.html). 
In this version only the tables BCZ_HydroState_V1 and WAM_ECMWF are implemented.

#### Sea Surface
table: BCZ_HydroState_V1
    * surface_baroclinic_eastward_sea_water_velocity
    * surface_baroclinic_northward_sea_water_velocity
    * sea_surface_height_above_sea_level
    * sea_surface_salinity
    * sea_surface_temperature

#### Wave Data
table: WAM_ECMWF
Output columns:
    * hs: wave height in cm
    * tm_1: wave period
    

### Time Data 
Data Related to time series. It adds the time of the day (day, night, twilight dawn...) and the moon phase. 
The calculation is done using skyfield (https://rhodesmill.org/skyfield/). 

### Csv Data 
Static data that is stored in a csv, with a lat and a lon columns (names to be given).
It returns the closest point of all the csv, the distance to it, the coordinates and also other columns selected by the
user with the specified suffix.

#### Shipwreck Data
Will add information about the closest shipwreck. The shipwrecks can be found back in the excel 
('data/wrakkendatabank.csv') and are confined to the BPNS.
Following information will be added:
* Distance to closest shipwreck
* (x,y) coordinates of shipwreck
* Name of shipwreck

## Usage 
Possible ways of loading the data 
```pyhton 
import bpnsdata 
import pandas as pd 
import numpy as np

# When the desired df is already on a gpx or a csv with coordinates (in this case, imagine the gpx itself 
# contains the rows to analyze
geofile = 'data/VG.gpx'
geodf = bpnsdata.SurveyLocation(geofile).geotrackpoints

# Could also be done directly using geopandas: 
geodf = geopandas.read_file(geofile)
gedf = geodf.set_index(pd.to_datetime(geodf['time']))

# Could be that we have a df (here a random one) and we want to add a geolocation to it
# Create a random dataframe to work with
time_index = pd.date_range(start='2020-10-12 11:35', end='2020-10-12 12:00', freq='m', tz='UTC')
random_data = np.random.randint(5, 30, size=10)
df = pd.DataFrame(random_data, index=time_index)
```
Use of the SeaDataManager
```python 
import bpnsdata

# Define the seadatamanager
env_vars = ['shipping', 'shipwreck', 'time', 'shipwreck', 'habitat_suitability',
            'seabed_habitat', 'sea_surface', 'sea_wave']
manager = bpnsdata.SeaDataManager(env_vars)

# If the data is without geometry, then:
geodf = manager.add_geodata(df, gpx_file)

# Once the data has geometry:
df_env = manager(geodf)
```

Use without the SeaDataManager
```python 
import bpnsdata

# For example, for shipping: 
shipping = bpnsdata.ShippingData(layer_name='rd', boat_type='All')
df_env = shipping(geodf)
```
