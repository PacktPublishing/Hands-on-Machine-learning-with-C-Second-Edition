cmake_minimum_required(VERSION 3.0)
project(blaze_samples)

set(BLAZE_PATH "" CACHE PATH "Path to Blaze")

if(NOT BLAZE_PATH)
  message(FATAL_ERROR "Missing Blaze install path, please specify BLAZE_PATH")
else()
  message("Blaze path is ${BLAZE_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${BLAZE_PATH})

add_executable(linalg_blaze "linalg_blaze.cc")
add_executable(linreg_blaze "linreg_blaze.cc")
target_link_libraries(linreg_blaze blas lapack)

