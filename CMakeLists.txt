cmake_minimum_required(VERSION 2.8.3)
project(addtimestamp)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(CMAKE_CXX_STANDARD 11)

# Find OpenMP
IF (APPLE)
    message(STATUS "clang does not support OpenMP!")
ELSEIF (WIN32 OR UNIX)
    find_package(OpenMP REQUIRED)
    IF(OPENMP_FOUND)
        message(STATUS "OpenMP Found!")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF()
ENDIF ()

# Libkdtree source files
set(SOURCE_FILES include/addtimestamp/kdtree.cpp include/addtimestamp/kdtree.h)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
)

find_package(PCL REQUIRED)

#find_package(Boost REQUIRED)

#include_directories(${Boost_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})

# Build libkdtree
add_library(kdtree SHARED ${SOURCE_FILES})

set_target_properties(kdtree PROPERTIES LINKER_LANGUAGE C)

add_executable(addtimestamp src/addtimestamp.cpp)
target_link_libraries(addtimestamp ${catkin_LIBRARIES} ${PCL_LIBRARIES} kdtree)


