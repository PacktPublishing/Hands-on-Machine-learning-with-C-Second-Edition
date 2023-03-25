cmake_minimum_required(VERSION 3.0)
project(arrayfire_samples)

find_package(ArrayFire)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${BLAZE_PATH})

add_executable(linalg_af "linalg_af.cc")
target_link_libraries(linalg_af ArrayFire::afcpu)

add_executable(linreg_af "linreg_af.cc")
target_link_libraries(linreg_af ArrayFire::afcpu)

