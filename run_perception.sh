#!/bin/bash
# Limpa qualquer contaminação do Noetic
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH

# Carrega só Foxy
source /opt/ros/foxy/setup.bash
source ~/perception_ws/install/setup.bash

# Executa o perception
ros2 run perception processing
