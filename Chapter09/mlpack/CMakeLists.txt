cmake_minimum_required(VERSION 3.10)
project(mlpack-ensemble)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(mlpack 4.0.1 REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
       -Wall -Wextra -msse3 -fopenmp
       $<$<CONFIG:RELEASE>:-Ofast>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)

add_link_options(-fopenmp)

add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
)

include_directories(${MLPACK_INCLUDE_DIR})

add_executable(mlpack-ensemble mlpack_ensemble.cc stacking.cc)
target_link_directories(mlpack-ensemble PRIVATE ${CMAKE_PREFIX_PATH}/lib)
target_link_libraries(mlpack-ensemble ${MLPACK_LIBRARIES} armadillo)
