#!bin/bash

# python3 Initialisation/EnvInitialisation.py
# python3 Pre_survey/Pre_surveyor.py
# python3 Pre_survey/Pre_surveyor.py > /dev/null 2>&1 &
# echo "Pre survey is finished"

python3 Pre_survey/PostProcessor.py > /dev/null 2>&1 &
# echo "Post processing is finished!"

python3 Pre_survey/PolygonHandler.py > /dev/null 2>&1 &
# echo "Polygon is generated successfully!"

python3 Pre_survey/GridHandler.py > /dev/null 2>&1 &
# echo "Grid is generated successfully!"

python3 Adaptive/PreProcessor.py > /dev/null 2>&1 &
# echo "Preprocessing is done! Ready to go with adaptive mission"

python3 Adaptive/MASCOT.py > /dev/null 2>&1 &
# echo "Mission Complete!"
