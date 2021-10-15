# The folder is cleaned

# Existing scripts
## Grid.py --> build grid for the mission
## GP.py --> build gaussian kernel for the mission
## Prior.py --> build prior for the mission
## test.py --> test script (workable / non-workable: ROS or not)
## Simulator.py --> purely testing the path planning
## func_test.py --> test random functions

# Existing folder
## backup --> including all the old grid and old prior scripts
## Polygon --> for testing onboard

### Notes
- nan needs to be filtered out before it is fed into the average system
- depth needs to be negative in Delft3D data to be able to compute
- selected lat and lon need to use grid, not from data, otherwise it will have singularity problem
- singular in distance matrix, may not cause singular in covariance matrix, since it can differ becasue of the exponential item

- empty eibv
- possibly get stuck, long term overshoot
- inside auvhander.setwaypoint, it needs to be positive for the depth component and speed component
