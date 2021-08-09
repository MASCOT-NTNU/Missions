Grid
- give origin, alpha, number of points, distance
- return: coordinates

GP
- set up covariance matrix
- set up parameters, iid noise

Prior
- set up prior

AUV
- set up the AUV

Path planner
- find starting loc
- find candidates
- find next according to some criterion


ROS ====

Data assimlator
- communicate with the robot
- assimilate data, acquire data

ROS Execution
- communicate with GP to update the field
- communicate with path planner to control the robot to move to the next waypoint

ROS ====


