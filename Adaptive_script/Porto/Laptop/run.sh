#!bin/bash

echo "Ready to take off"

python3 Polygon_Selector.py
python3 Prior.py
python3 Path_designer_Initial_survey.py
python3 GP.py

say "Congrats, all are set up, ready to take off, good luck. "
