cmake_minimum_required(VERSION 3.22)
project(eigen_samples)

find_package(OpenMP)
find_package(Eigen3 3.4.0 REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(
       -Wall -Wextra -msse3
       $<$<CONFIG:RELEASE>:-Ofast>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)

add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
)

add_executable(linalg_eigen "linalg_eigen.cc")
target_link_libraries(linalg_eigen Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(linreg_eigen "linreg_eigen.cc")
target_link_libraries(linreg_eigen Eigen3::Eigen OpenMP::OpenMP_CXX)
