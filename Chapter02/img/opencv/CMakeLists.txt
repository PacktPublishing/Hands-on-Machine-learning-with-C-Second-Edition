cmake_minimum_required(VERSION 3.22)
project(img-opencv)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(
       -Wall -Wextra -msse3 -fopenmp
       $<$<CONFIG:RELEASE>:-Ofast>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)

add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
)

add_executable(img-opencv "ocv.cc")
target_link_libraries(img-opencv  opencv_core opencv_imgproc opencv_highgui)
