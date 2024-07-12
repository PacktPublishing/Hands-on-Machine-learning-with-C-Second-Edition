cmake_minimum_required(VERSION 3.22)
project(lenet-pytorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)


include_directories(${TORCH_INCLUDE_DIRS})

add_compile_options(
       -Wall -Wextra -msse3 -fopenmp
       $<$<CONFIG:RELEASE>:-Ofast>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)

set(SOURCE_FILES main.cpp
                 mnistdataset.h
                 mnistdataset.cpp
                 lenet5.h
                 lenet5.cpp)

add_executable("${CMAKE_PROJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${TORCH_LIBRARIES} ${OpenCV_LIBS})

