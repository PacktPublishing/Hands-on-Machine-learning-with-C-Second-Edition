cmake_minimum_required(VERSION 3.22)
project(dlib-samples)

find_package(dlib 19.24 REQUIRED)

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

add_executable(linalg-dlib "linalg_dlib.cc")
target_link_libraries(linalg-dlib dlib::dlib)

add_executable(linreg-dlib "linreg_dlib.cc")
target_link_libraries(linreg-dlib dlib::dlib)

