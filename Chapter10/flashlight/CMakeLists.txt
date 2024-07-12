cmake_minimum_required(VERSION 3.10)
project(mlp-flashlight)

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
    mlp_fl.cc
    ../data/data.h
    ../data/data.cc
)

add_executable(mlp_fl ${SOURCES})
target_link_libraries(mlp_fl flashlight::flashlight)
