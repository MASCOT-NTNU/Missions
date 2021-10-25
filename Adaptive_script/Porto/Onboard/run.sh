#!bin/bash

python3 Initialisation/EnvInitialisation.py
# python3 Pre_survey/Pre_surveyor.py
python3 Pre_survey/Pre_surveyor.py > /dev/null 2>&1 &
# echo "Pre survey is finished"

python3 Adaptive/PreProcessor.py > /dev/null 2>&1 &
# echo "PreProcessing is finished!"

python3 Adaptive/MASCOT.py > /dev/null 2>&1 &
# echo "Mission Complete!"
