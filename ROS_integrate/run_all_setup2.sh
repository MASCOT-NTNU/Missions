#!/bin/bash

cd /home/yaoling/MASCOT/setup/
chmod +x setup_netpus.sh
chmod +x setup_dune.sh
chmod +x setup_ros1.sh
chmod +x setup_ros2.sh
parallel -u ::: './setup_ros1.sh 1' './setup_ros2.sh 2' 

