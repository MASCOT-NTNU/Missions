# Cores of the MASCOT autonomous ocean sampling

## TL;DR

###### Laptop folder needs to be done on laptop, which will generate files needed in Onboard.
###### Onboard folder contains all necessary files needed for the mission.

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
