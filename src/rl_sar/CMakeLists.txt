cmake_minimum_required(VERSION 3.0.2)
project(rl_sar)

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")
find_package(gazebo REQUIRED)

find_package(TBB REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  controller_manager
  genmsg
  joint_state_controller
  robot_state_publisher
  roscpp
  gazebo_ros
  std_msgs
  tf
  geometry_msgs
  robot_msgs
  robot_joint_controller
  rospy
  urdf
  gazebo_msgs
  visualization_msgs
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 添加OpenCV
find_package(OpenCV REQUIRED)

# 确保包含 yaml-cpp
find_package(yaml-cpp REQUIRED)

catkin_package(
  CATKIN_DEPENDS
  robot_joint_controller
  rospy
  tf2_ros
  geometry_msgs
  tf
)

link_directories(/usr/local/lib)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Unitree A1
include_directories(library/unitree_legged_sdk_3.2/include)
link_directories(library/unitree_legged_sdk_3.2/lib)
set(UNITREE_A1_LIBS -pthread unitree_legged_sdk_amd64 lcm)

# Unitree Go2
include_directories(library/unitree_sdk2/include)
link_directories(library/unitree_sdk2/lib/x86_64)
include_directories(library/unitree_sdk2/thirdparty/include)
include_directories(library/unitree_sdk2/thirdparty/include/ddscxx)
link_directories(library/unitree_sdk2/thirdparty/lib/x86_64)
set(UNITREE_GO2_LIBS -pthread unitree_sdk2 ddsc ddscxx)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  library/matplotlibcpp
  library/observation_buffer
  library/rl_sdk
  library/loop
)

add_library(rl_sdk
  library/rl_sdk/rl_sdk.cpp
  library/rl_sdk/bbox.cpp
)
target_link_libraries(rl_sdk
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${ASSIMP_LIBRARIES}
  yaml-cpp
  Python3::Python
  Python3::Module
  TBB::tbb
  ${OpenCV_LIBS}
  ${PYTHON_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)
set_property(TARGET rl_sdk PROPERTY CXX_STANDARD 17)
set_property(TARGET rl_sdk PROPERTY CXX_STANDARD_REQUIRED ON)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(rl_sdk Python3::NumPy)
else()
  target_compile_definitions(rl_sdk WITHOUT_NUMPY)
endif()

target_include_directories(rl_sdk PUBLIC
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
  ${YAML_CPP_INCLUDE_DIR}
)

add_library(observation_buffer library/observation_buffer/observation_buffer.cpp)
target_link_libraries(observation_buffer "${TORCH_LIBRARIES}")
set_property(TARGET observation_buffer PROPERTY CXX_STANDARD 14)

add_executable(rl_sim src/rl_sim.cpp )
target_link_libraries(rl_sim
  ${catkin_LIBRARIES}
  rl_sdk
  observation_buffer
  yaml-cpp
)

add_executable(rl_real_a1 src/rl_real_a1.cpp )
target_link_libraries(rl_real_a1
  ${catkin_LIBRARIES}
  ${UNITREE_A1_LIBS}
  rl_sdk
  observation_buffer
  yaml-cpp
)

add_executable(rl_real_go2 src/rl_real_go2.cpp )
target_link_libraries(rl_real_go2
  ${catkin_LIBRARIES}
  ${UNITREE_GO2_LIBS}
  rl_sdk
  observation_buffer
  yaml-cpp
)

add_executable(tf_relay src/tf_relay.cpp)
target_link_libraries(tf_relay
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/rl_sim.py
  scripts/actuator_net.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 添加编译选项
add_compile_options(-std=c++17)