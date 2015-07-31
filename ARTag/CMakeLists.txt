cmake_minimum_required(VERSION 2.8)
project( ARTagDetector )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O4")
FIND_PACKAGE( OpenCV REQUIRED )

add_executable( ARTagDetector main.cpp)
target_link_libraries( ARTagDetector ${OpenCV_LIBS})
