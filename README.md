# MASCOT (Maritime Autonomous Ocean Sampling)

The algorithm is built on the adaptive sampling effort from Trygve's previous work.

The main idea behind the current system is to conduct the adaptive sampling in the Rio Douro with only taking the salinity difference into account.

One 2D simulation case is shown in ![2D simulation](Porto/Setup/Simulation/fig/P22/test.mp4)

<!-- One 3D simulation case is shown in ![3D simulation](Porto/Setup/Grid/fig/P1/test.mp4) -->

The system consists of 4 modules:
- Grid
- GaussianProcessKernel
- Prior
- AUV-Runner

---
Here comes more detailed introduction of each module.

---
## Grid module

The polygon grid discretisation is designed for this specific application, given that the computing platform is only capable of manipulating the limited number of grid nodes in the waypoint graph. Therefore the pre-inspected survey can be used to select the desired polygon which most likely contains the front of the river plume. The example of the polygon grid discretisation can be found here ![Dynamic waypoint generation](Porto/Setup/Grid/fig/P1/test.mp4).


## GaussianProcessKernel module

The gaussian process kernel module is used to specify the parameters for the entire algorithm, which needs to be adjusted before the actual deployment to achieve the optimal desired performance. The initial survey using AUV for sampling transect lines or yoyo patterns can help to adjust the threshold, sigma, eta and tau in the kernel based on the spatial variogram.

## Prior module

The objective of the prior module is to build a prior mean, which can then be used for the starting point of the adaptive exploration.

Given the current setup, two alternative prior can be used for comparison.

- Prior I: wind-direction-oriented prior based on the past data.
  Input:
  - grid coordinates.
  - numerical data.

  Output:
  - selected regional data with the wind-direction-oriented prior.

- Prior II: forcasted data on the mission day.
  Input:
  - grid discretisation.
  - numerical data.

  Output:
  - selected regional data based on the forcasted model.

## AUV-Runner module

The objective of this module is to communicate with ROS and achieve of controlling function by using the adaptive sampling framework developed by the AURLab at NTNU.

It achieves the adaptive sampling by setting the next waypoint to be the one which has the minimum EIBV (Expected Integrated Bernoulli Variance).


---

# Contact

Please contact Yaolin Ge (yaolin.ge@ntnu.no) if you have any questions. 😁 🤔 🤘
