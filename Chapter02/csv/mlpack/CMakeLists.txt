cmake_minimum_required(VERSION 3.0)
project(mlpack-csv-sample)

include(FindPackageHandleStandardArgs)

find_package(OpenMP REQUIRED)

find_path(MLPACK_INCLUDE_DIR
    NAMES mlpack/core.hpp mlpack/prereqs.hpp
)

find_package_handle_standard_args(mlpack
    REQUIRED_VARS MLPACK_INCLUDE_DIR
)

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

include_directories(${MLPACK_INCLUDE_DIR})

add_executable(csv_mlpack csv_mlpack.cc)
target_link_libraries(csv_mlpack OpenMP::OpenMP_CXX)