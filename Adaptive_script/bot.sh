#!bin/bash

rm -rf ROS_PreRun.py
rm -rf usr_func.py

wget https://raw.githubusercontent.com/MASCOT-NTNU/Missions/master/Adaptive_script/ROS_PreRun.py
wget https://raw.githubusercontent.com/MASCOT-NTNU/Missions/master/Adaptive_script/usr_func.py

cd ../../../
catkin_make
source devel/setup.bash

cd src/adaframe_examples/scripts/

chmod +x ROS_PreRun.py
chmod +x usr_func.py
