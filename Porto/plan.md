
- Depth is limited to be within 15m, so 3 layers are not decided to be conducted.
- How long will be the whole test, it is determined by the AUV service life
- How to deal with the pop-up issue, together with the 3D path planning, since it needs to be accurate in terms of its own navigation
- LSTS: It could make sense to include an UAV to help on the first search of the plume. Do you think is feasible?
- LSTS: Do we have the possibility of synchronising our mission with satellite Sentinel-2 and Landsat over passing in the area?
- LSTS: What is the computation footprint the the algorithm has?
- LSTS: Is it possible to run on a RASP4?
- LSTS: This being a backseat, is it prepare just to install it an run it? or integration work is needed to be done on our side?


There are some responses to your questions. 
- What type of sensors do UAV have, can it provide high-resolution images? If the boundary cannot be visibly distinguished from the air, does the UAV have any other sensor to carry out this work?

- What do you mean by synchronising the mission with satellite. We will possibly use both Delft-3D and Sentinel-2 for building some math models and then surveying the fields using AUVs to update the model only based on the measurements from sensors on AUV, so before the deployment, we can use as much data as possible, once it is deployed, no interaction (at least for now) between AUV and us will be allowed, so called autonomous sampling. However, this can be a possible solution for later questions

- The computational cost is quite high for now, it can be dramatically reduced by using some complex libraries which might need to be additionally installed. We are currently working on this now, lots of simulations are being conducted to improve the performance of the algorithm. 

- I am not so sure about the computational capacity on RASP4. But we will run our algorithm on LAUV-Harald from NTNU AURLab, so I believe you know the difference between those two. 

- We are developing some ROS-Dune simulator developed by AURLab at NTNU, which can then reduce the integration time. As long as it is able to be run from this simulator, then it should be ready on the AUV as well. At least, it will work on AUVs from the AURLab, but I am not sure if you have the same setting with them. 

But I will consult more people and get back to you as soon as possible. 

