#SOUNDEXPLORER 

Soundexplorer is a pakage to ease acoustic data exploration. 
The idea is to be able to process several deployments only with one command. 
All the deployments have to be listed in a csv file with all the metadata, an example
is provided in docs/data_summary_example.csv.
The main classes of the soundexplorer are Dataset and Deployment. 
A deployment represents a period of data acquisition with constant metadata (same instrument and 
instrument settings). A Dataset is a conjunction of deployments to be studied together.

To process the acoustic data, pypam is used . It can be called by using the function generate_deployment_data 
from the Deployment class. 
A part from acoustic processing, environmental data can be added. 
Right now only Belgian Part of the North Sea data is avalable for al lthe classes. 
They are: 

## Human Data
Shipping activity from https://www.emodnet-humanactivities.eu/
Adds the route density or the shipping intensity from the month of the deployment to the dataset, considering the 
location 

## Map Data 
Map Data represents geographical data. The three outputs are: 
bathymetry (https://www.emodnet-bathymetry.eu/), sea habitat (https://www.emodnet-seabedhabitats.eu/), and 
benthic habitats ([1]V. Van Lancker, G. Moerkerke, I. Du Four, E. Verfaillie, M. Rabaut, and S. Degraer, “Fine-scale Geomorphological Mapping of Sandbank Environments for the Prediction of Macrobenthic Occurences, Belgian Part of the North Sea,” Seafloor Geomorphology as Benthic Habitat, pp. 251–260, 2012, doi: 10.1016/B978-0-12-385140-6.00014-1.
). The closest point from the maps is added to the each point of the dataset 


## Sea State Data
Sea State Data from RBINS (https://erddap.naturalsciences.be/erddap/index.html). 
from the tables BCZ_HydroState_V1 and WAM_ECMWF. 
It computes for each row: 
* BCZ_HydroState_V1
    * surface_baroclinic_eastward_sea_water_velocity
    * surface_baroclinic_northward_sea_water_velocity
    * sea_surface_height_above_sea_level
    * sea_surface_salinity
    * sea_surface_temperature
* WAM_ECMWF
    * hs
    * tm_1
    

## Time Data 
Data Related to time series. It adds the time of the day (day, night, twilight dawn...) and the moon phase. 
The calculation is done using skyfield (https://rhodesmill.org/skyfield/). 

## Shipwreck Data
Will add information about the closest shipwreck. The shipwrecks can be found back in the excel ('data/wrakkendatabank.afdelingkust.be-json-export_2021-04-02.xls') and are confined to the BPNS (belgian part of the north sea).
Following information will be added:
* Distance to closest shipwreck
* (x,y) coordinates of shipwreck
* Name of shipwreck

