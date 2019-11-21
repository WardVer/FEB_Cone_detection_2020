cmake_minimum_required(VERSION 3.8)

project(FEB_Cone_detection_2020 VERSION 0.0.1)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package( OpenCV REQUIRED )
link_directories(./libraries)
include_directories(./include)

add_executable(Cone_detector ./src/detect_cone.cpp)

Target_link_libraries(Cone_detector dark ${OpenCV_LIBS})

