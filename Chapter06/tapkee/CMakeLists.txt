cmake_minimum_required(VERSION 3.10)
project(tapkee-dr)

find_package(OpenMP)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(fmt)

set(TAPKEE_PATH "" CACHE PATH "path to tapkee install dir")
if (NOT TAPKEE_PATH)
  message(FATAL_ERROR "Missing tapkee include path, please specify TAPKEE_PATH")
else()
  message("tapkee path is ${TAPKEE_PATH}")
endif()

set(PLOTCPP_PATH "" CACHE PATH "path to poltcpp install dir")
if (NOT PLOTCPP_PATH)
  message(FATAL_ERROR "Missing plotcpp include path, please specify PLOTCPP_PATH")
else()
  message("plotcpp path is ${PLOTCPP_PATH}")
endif()


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

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
include_directories(${TAPKEE_PATH})

add_executable(tapkee-dr tapkee-dr.cc util.cc)
target_link_libraries (tapkee-dr Eigen3::Eigen OpenMP::OpenMP_CXX fmt::fmt)
