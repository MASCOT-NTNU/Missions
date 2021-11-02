scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/* pi@10.0.2.32:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/


scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/*.py pi@10.0.2.32:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/


scp -r pi@10.0.2.32:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/Data/* /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Data/Porto/Simulator

scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/*.kml pi@10.0.10.123:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/

scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/*.txt pi@10.0.10.123:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/

scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/*.py pi@10.0.10.123:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/

scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/MASCOT.py pi@10.0.10.123:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/

scp /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/*.py pi@10.0.10.123:/scp pi@10.0.10.123:/home/pi/adaframe_ws/src/imc_ros_interface/scripts/auv_handler.py .

/home/pi/adaframe_ws/src/imc_ros_interface/scripts

ip: 10.0.2.33
netmask: 255.255.0.0
router: 10.0.0.1

pi@10.0.10.123
horsetomato


scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Pre_survey/Pre_surveyor.py pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/
scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Pre_survey/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/Pre_survey*

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/Config/

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Adaptive/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT_PORTO/Adaptive/

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/path_initial_survey.txt pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Config/
scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Adaptive/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Adaptive/

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/run*.sh pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/

scp -r /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Data/Prior/* pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Data/Prior/

scp -r pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Data/Pre_survey/Pre_survey_data_on_2021_1026_1231/* .
scp -r pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Data/MASCOT/MissionData_on_2021_1026_1614/* .

scp -r pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/ /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/Porto/Simulation/20211027/

scp -r pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Data/Corrected/* .
scp -r pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Config/* .
scp -r pi@129.241.15.208:/home/pi/adaframe_ws/src/adaframe_examples/scripts/MASCOT/Data/

https://studntnu-my.sharepoint.com/:u:/g/personal/yaoling_ntnu_no/Ebbkh6lXY1JGkpVPlx5QwpYB0xaw6-2AZPuzAfyDd6rWFg?e=cgnOaI

git init
git remote add origin https://github.com/MASCOT-NTNU/Missions.git

git config core.sparseCheckout true
echo "https://github.com/MASCOT-NTNU/Missions/tree/master/Adaptive_script/Porto/Onboard/" > .git/info/sparse-checkout

git pull origin master


svn checkout https://github.com/MASCOT-NTNU/Missions/trunk/Adaptive_script/Porto/Onboard
