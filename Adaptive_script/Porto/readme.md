# Porto folder contains essential scripts and data for mission execution

# It is composed of two major components such as `Laptop` and `Onboard`.

# `Laptop` needs to be run on the usr laptop before copying everything from `Onboard` to the AUV.


### Notes
- nan needs to be filtered out before it is fed into the average system
- depth needs to be negative in Delft3D data to be able to compute
- selected lat and lon need to use grid, not from data, otherwise it will have singularity problem
- singular in distance matrix, may not cause singular in covariance matrix, since it can differ becasue of the exponential item

- empty eibv
- possibly get stuck, long term overshoot
- inside auvhander.setwaypoint, it needs to be positive for the depth component and speed component
