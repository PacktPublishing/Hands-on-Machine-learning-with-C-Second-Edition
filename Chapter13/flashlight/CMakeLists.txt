cmake_minimum_required(VERSION 3.10)
project(fl_track)

find_package(flashlight 0.4.0 REQUIRED)

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

include_directories(${PLOTCPP_PATH})

set(SOURCES
    fl_track.cc
    mlflow.cc
)

add_executable(fl_track ${SOURCES})
target_link_libraries(fl_track flashlight::flashlight)
