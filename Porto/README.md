# Cores of the MASCOT autonomous ocean sampling

## TL;DR

###### `Laptop` folder needs to be done on laptop, which will generate files needed in Onboard. I will take care of this part. You don't have to worry about this. Changes for those parameters and polygon might change depend on the weather conditions, forcasted models, satellite images and Drone footage.
###### `Onboard` folder contains all necessary files needed for the mission. Only this folder will be transferred to `LAUV-XPLORE-1`. Only this folder will be used during the mission onboard.

---
# Laptop:
Copy in the command line: `sh run.sh`, this will execute all essential scripts including:
- `Polygon_Selector.py` pops up window for selecting the proper polygon and generates the grid inside.
- `Prior.py` extract data from Delft3D onto those grid points.
- `GP.py` initialises the Gaussian Process Kernel.
- `Path_designer_initial_survey.py` pops up another window for selecting the desired path for the initial survey.

**Args need to be changed according to the wind direction and the wind level, also the tide window for ebb phase. Last but not least, the correct mission day. One example of north moderate wind with high tide from 8 to low tide at 12 on Sept 14th is listed below:
```
sh run.sh
pycharm = False, use terminal instead.
string_date = "2021-09-14_2021-09-15"
string_hour = "08_12"
wind_dir = "North" # [North, East, West, South]
wind_level = "Moderate" # [Mild, Moderate, Heavy]
```

# Onboard:
First all the files created inside Onboard folder needs to be transferred to the AUV.
Then run in the command line: `sh run.sh`

This will run two scripts inside the folder: `Pre_surveyor.py` and `MASCOT.py`

`Pre_surveyor.py` actuates the AUV to sample with the initial path which is defined during the laptop phase, i.e., `path_initial_survey.txt` contains all waypoints which will be used to sample the field. They are predefined. After the AUV has sampled all of those waypoints. It will stay on the surface and use that data to adjust the Delft3D data. Then the corrected Delft3D data will be used as the prior for the next stage autonomous sampling using script `MASCOT.py`

`MASCOT.py` actuates the AUV to sample the field with EIBV as its path planning algorithm, i.e., the goal of this script is to find the boundary between the river plume and the ocean. Its sharpness depends on the threshold and some parameters inside `GP.py`. The simulation from the main page has shown where it might go.

---
#### Behaviours of the adaptive mission (common features):
---
1. Data will be saved every 10 seconds. `data = np.array([[lat, lon, depth, salinity]])`
2. In `Pre_surveyor.py`, yoyo pattern is used, i.e., the AUV moves itself between 0m (surface) and 2m. The data will be merged to the depth layer [0.5, 1.25, 2.0]m for the adaptive mission. Every other waypoint will be on the surface, i.e., the AUV starts from the surface (the first waypoint), moves down to 2m as its second waypoint, then pops up to the surface. Everytime when it pops up, it will stay on the surface for 30 seconds. However, it will stay on the surface for 90 seconds after every 10 mins.
3. In `MASCOT.py`, adaptive sampling in full scale 3D is implemented, i.e., the AUV moves towards the location with minimum EIBV (where the boundary might exist). Every other 5 waypoints (approx. 7 mins), it will pop up and stay on the surface for 30 seconds, every other 10 mins, it will pop up and stay on the surface on 90 seconds. Later on, it will dive to the next location if it is needed.
4. Everytime it surfaces, it will send a SMS after 30 seconds, i.e., if it only stays on the surface for 30 seconds, it will only send one SMS. However, if it stays on the surface for 90 seconds, it will send 3 SMSs. Be aware that not every SMS sent by the AUV can be received due to bad conditions.
