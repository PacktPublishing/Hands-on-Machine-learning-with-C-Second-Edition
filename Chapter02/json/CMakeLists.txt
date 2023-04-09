cmake_minimum_required(VERSION 3.22)
project(json_sample)

find_package(nlohmann_json 3.11.2 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

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

set(SOURCES json.cc
            review.h
            paper.h
            reviewsreader.h
            reviewsreader.cpp)

add_executable(json_sample ${SOURCES})
target_link_libraries(json_sample Eigen3::Eigen nlohmann_json::nlohmann_json)
