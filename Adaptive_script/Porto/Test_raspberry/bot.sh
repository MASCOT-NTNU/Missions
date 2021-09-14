#!bin/bash

cd ../../../../
catkin_make
source devel/setup.bash

cd src/adaframe_examples/scripts/PolyGon/

chmod +x GP.py
chmod +x Grid.py
chmod +x Prior.py
chmod +x test.py
chmod +x Data_analysis.py
