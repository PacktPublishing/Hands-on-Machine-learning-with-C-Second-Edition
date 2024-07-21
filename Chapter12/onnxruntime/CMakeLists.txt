cmake_minimum_required(VERSION 3.10)
project(onnx-load)

find_package(onnxruntime REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs)

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
    main.cc
)

add_executable(onnx-load ${SOURCES})
target_link_libraries(onnx-load onnxruntime::onnxruntime opencv_core opencv_imgproc opencv_imgcodecs)
