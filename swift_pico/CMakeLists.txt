cmake_minimum_required(VERSION 3.8)
project(swift_pico)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)               #ROS2 C++ client library
find_package(rclpy REQUIRED)                #ROS2 Python client library
find_package(swift_msgs REQUIRED)           #Drone command msg
find_package(controller_msg REQUIRED)       #PID Tuning msg
find_package(error_msg REQUIRED)            #Error visualization msg
find_package(whycode_interfaces REQUIRED)   #WhyCode interfaces
find_package(geometry_msgs REQUIRED)        #Common ROS2 message types
find_package(OpenCV REQUIRED)

install(
  PROGRAMS
    config/ros2_bridge.py
  RENAME mujoco_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    config/motor_dynamics.py
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()