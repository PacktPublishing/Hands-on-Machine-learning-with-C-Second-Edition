cmake_minimum_required(VERSION 3.0)
project(flashlight-csv-sample)

set(CSV_LIB_PATH "" CACHE PATH "Path to csv library include dir")

if (NOT CSV_LIB_PATH)
  message(FATAL_ERROR "Missigng CSV lib install path, please specify CSV_LIB_PATH")
else()
  message("CSV lib path is ${CSV_LIB_PATH}")
endif()

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

include_directories(${CSV_LIB_PATH})

add_executable(csv_fl csv_fl.cc)
target_link_libraries(csv_fl flashlight::flashlight)
