#!bin/bash

cd ../../../../
catkin_make
source devel/setup.bash

cd src/adaframe_examples/scripts/MASCOT_PORTO/

chmod +x Pre_surveyor.py
chmod +x MASCOT.py
