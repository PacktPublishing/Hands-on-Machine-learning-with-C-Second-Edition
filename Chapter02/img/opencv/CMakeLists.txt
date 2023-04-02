cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(opencv-example)

find_package(OpenCV REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

add_compile_options( -fopenmp
                     -pthread
                     -msse3
                     -Wall
                     -Wextra
                     -Wno-unused-parameter)

add_executable(ocv_sample ocv.cc)
target_link_libraries(ocv_sample ${OpenCV_LIBS})
