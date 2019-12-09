cmake_minimum_required(VERSION 3.8)

project(FEB_Cone_detection_2020 VERSION 0.0.1)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package( OpenCV REQUIRED )
link_directories(./libraries usr/lib)
include_directories(./include)

add_library(
    3d
    ./src/3d_functions.cpp
    ./include/3d_functions.hpp
)

add_executable(Cone_detector ./src/detect_cone.cpp)
add_executable(calibrate ./src/calibrate.cpp)
add_executable(calibrate_pics ./src/calibrate_pics.cpp)
add_executable(camera_test ./src/camera_test.cpp)

Target_link_libraries(3d ${OpenCV_LIBS})
Target_link_libraries(Cone_detector dark ${OpenCV_LIBS} 3d gxiapi)
Target_link_libraries(calibrate ${OpenCV_LIBS})
Target_link_libraries(calibrate_pics ${OpenCV_LIBS} gxiapi)
Target_link_libraries(camera_test ${OpenCV_LIBS} gxiapi)


