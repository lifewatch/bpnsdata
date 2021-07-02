# SOUNDEXPLORER 

Soundexplorer is a package to ease acoustic data exploration. 
The idea is to be able to process several deployments only with one command. 
All the deployments have to be listed in a csv file with all the metadata, an example
is provided in docs/data_summary_example.csv.
The main classes of the soundexplorer are Dataset and Deployment. 
A deployment represents a period of data acquisition with constant metadata (same instrument and 
instrument settings). A Dataset is a conjunction of deployments to be studied together.
The output is always in a structured folder.
* output_folder/
    * deployments/: contains one pickle file per deployment processed
    * detections/: contains the output of the detectors (if applicable)
    * img/: contains graphs created (if applicable)
        * data_overview/: general plots 
        * features_analysis/: stats from the features
        * temporal_features/: graphs in time domain of the features 
        * spatial_features/: spatial plots (in a map) of the features
    * dataset.pkl : dataset containing all the deployments from deployments/
    

## Acoustic data and general process
To process the acoustic data, pypam (https://github.com/lifewatch/pypam) is used. 
It can be called by using the function generate_deployment_data from the Deployment class to generate the data from one 
deployment, or generate_entire_dataset when multiple deployments have to be processed. 
When multiple deployments have to be processed, the metadata has to be summarized in a csv file. Each row is a 
deployment, and an example can be found at docs/data_summary_example.csv.
This metadata information will be at one point linked with the download output of ETN Underwater Acoustics 
(https://www.lifewatch.be/etn/).

The output of the data generation is 

## Environmental data
A part from acoustic processing, environmental data can be added by specifying it in the env_vars
variable when calling the generate_entire_dataset. To do so, it is necessary to have gps information, which 
should be stored in a .gpx file, in the "waypoints" or "track_points" layer. Then the algorithm finds the point
which is closest in time for each processed acoustic bin.
The available variables are: 
* spatial_data
* sea_state
* time_data
* sea_bottom
* shipping
* shipwreck

To add environmental variables, it is necessary to include 'spatial_data', which will read the gps information 
and correlate it with the sound. 

Right now only Belgian Part of the North Sea data is available for all the classes. 
They are: 

### Human Data
Shipping activity from https://www.emodnet-humanactivities.eu/
Adds the route density or the shipping intensity from the month of the deployment to the dataset, considering the 
location 

### Map Data 
Map Data represents geographical data. The three outputs are: 
bathymetry (https://www.emodnet-bathymetry.eu/), sea habitat (https://www.emodnet-seabedhabitats.eu/), and 
benthic habitats ([1]V. Van Lancker, G. Moerkerke, I. Du Four, E. Verfaillie, M. Rabaut, and S. Degraer, “Fine-scale Geomorphological Mapping of Sandbank Environments for the Prediction of Macrobenthic Occurences, Belgian Part of the North Sea,” Seafloor Geomorphology as Benthic Habitat, pp. 251–260, 2012, doi: 10.1016/B978-0-12-385140-6.00014-1.
). The closest point from the maps is added to the each point of the dataset 


### Sea State Data
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
    

### Time Data 
Data Related to time series. It adds the time of the day (day, night, twilight dawn...) and the moon phase. 
The calculation is done using skyfield (https://rhodesmill.org/skyfield/). 

### Shipwreck Data
Will add information about the closest shipwreck. The shipwrecks can be found back in the excel 
('data/wrakkendatabank.afdelingkust.be-json-export_2021-04-02.xls') and are confined to the BPNS 
(belgian part of the north sea).
Following information will be added:
* Distance to closest shipwreck
* (x,y) coordinates of shipwreck
* Name of shipwreck


## Use example    
    # Acoustic Data
    summary_path = pathlib.Path('docs/data_summary_example.csv')
    include_dirs = False
    
    # Output folder
    output_folder = summary_path.parent.joinpath('data_exploration')
    
    # Hydrophone Setup
    # If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
    model = 'ST300HF'
    name = 'SoundTrap'
    serial_number = 67416073
    soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
    
    # All the instruments from the dataset can be added in the dictionary
    instruments = {'SoundTrap': soundtrap}
    
    # Acoustic params. Reference pressure 1 uPa
    REF_PRESSURE = 1e-6
    
    # SURVEY PARAMETERS. Look into pypam documentation for further information
    nfft = 4096
    binsize = 5.0
    band_lf = [50, 500]
    band_mf = [500, 2000]
    band_hf = [2000, 20000]
    band_list = [band_lf, band_mf, band_hf]
    
    # Features can be any of the features that can be passed to pypam
    features = ['rms', 'sel', 'aci']
    
    # Third octaves can be None (for broadband analysis), a specific band [low_freq, high_freq], for 
    only certain band analysis or False if no computation is wanted
    third_octaves = False
    
    env_vars = ['spatial_data', 'sea_state', 'time_data', 'sea_bottom', 'shipping', 'shipwreck']
    
    # Generate the dataset
    dataset = dataset.DataSet(summary_path, output_folder, instruments, features, third_octaves, band_list, binsize, nfft)
    dataset.generate_entire_dataset(env_vars=env_vars)
 

