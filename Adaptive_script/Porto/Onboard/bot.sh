#!bin/bash

cd ../../../
catkin_make
source devel/setup.bash

cd src/adaframe_examples/scripts/

chmod +x *

#chmod +x MASCOT.py
