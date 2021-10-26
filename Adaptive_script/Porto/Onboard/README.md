# Autonomous Ocean Sampling with LAUV

This repo contains all the essential files for conducting the autonomous ocean sampling with LAUVs.

---

## Mission Prepration
- Copy all the files from [REPO](https://github.com/MASCOT-NTNU/Missions/tree/master/Adaptive_script/Porto/Onboard) to the script folder which contains Tore's code

<!-- - Next run `Config.sh` to configure the essential paths -->
- Create a new tab from terminal, run the following commands inside adaframe_ws/:

```bash

cd adaframe_ws/
source devel/setup.bash
cd src

roslaunch imc_ros_interface/launch/bridge.launch
```

- Create a new tab and navigate into `/adaframe_ws/src/adaframe_examples/scripts/MASCOT/`

---

## Mission launching

- Launch the PreSurvey by `sh run1.sh`

- Launch the PostProcessing and PreProcessing by `sh run2.sh`

- Launch the adaptive mission by `sh run3.sh`


<!-- - If the mission is aborted due to some reasons such as boat traffic or high waves etc., then one can run `sh run_resume.sh` to continue with the mission without starting from the beginning. -->


---

# Contact

Please contact Yaolin Ge (yaolin.ge@ntnu.no) if you have any questions. 😁 🤔 🤘
