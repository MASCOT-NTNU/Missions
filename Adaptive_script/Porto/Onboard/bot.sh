#!bin/bash

cd ../../../../
catkin_make
source devel/setup.bash

cd src/adaframe_examples/scripts/MASCOT_PORTO/

chmod +x Pre_survey/Pre_surveyor.py
chmod +x Pre_survey/AUV.py
chmod +x Pre_survey/DataHandler.py
chmod +x Pre_survey/MessageHandler.py
chmod +x Pre_survey/PostProcessor.py
chmod +x Pre_survey/usr_func.py

#chmod +x MASCOT.py
