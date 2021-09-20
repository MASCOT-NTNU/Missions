# Testing on the laptop:
## Path will be used:
path_map = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot"
path_grid = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/fig/"
path_maretec = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/"
path_delft3d = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/"

# Steps:

## Step I: For polygon selector, things need to be specified, select pycharm = False, use terminal to launch the script:

### New data needs to be loaded from MARETEC, which are updated during the mission day

python3 Polygon_Selector.py

pycharm = False, use terminal instead. 

string_date = "2021-09-14_2021-09-15"
string_hour = "08_12"
wind_dir = "North" # [North, East, West, South]
wind_level = "Moderate" # [Mild, Moderate, Heavy]

## Step II:


Done on my laptop:
run polygon selector, to select the polygon, threshold --> polygon.txt

Onboard:
threshold needs to be set based on the gradient.
initial survey on the desired path, --> grid.txt, path.txt
prior calibrated based on the actual measured data --> data.txt
Actual mission, data will be saved in a file --> data_mission.txt, in case of erasing, something needs to be thought over.


Plan I:
Prepare two polygons, for validation, different directions


run initial surveyor, it will create a data file contains the initial survey data
run mission, which calls prior to build the calibrated prior, specify the collected data path, since it needs to be autonomously, no human intervention, thus, the path needs to be thought of carefully.

Inside mission, the data file will be imported and gathered around its location, and have a linear regression

Onboard: contains all the files which will be transferred to the AUV,
Laptop: contains all the files which will be used for generating all necessary files in onboard folder


Laptop:
- Polygon_Selector
- Prior
- Pathdesigner_initialsurvey
- GP, so no need for onboard computation, will be saved to text file
- data analysis for both delft3d and auv measurements
- variogram along lateral and vertical direction

Step:
Run Polygon_Selector
Run Prior
Run GP
Run Pathdesigner_initialsurvey


Onboard:
- Pre_surveyor:
  input: path_initial_survey, path for the initial survey
  output: corrected prior, with lat, lon, depth, sal.
- MASCOT
  input: mu prior, sigma prior,
  output: waypoints which actuate the AUV
