cmake_minimum_required(VERSION 3.10)
project(dlib-img-sample)

set(DLIB_PATH "" CACHE PATH "path to dlib install dir")

if (NOT DLIB_PATH)
  message(FATAL_ERROR "Missigng Dlib install path, please specify DLIB_PATH")
else()
  message("Dlib path is ${DLIB_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(requiredlibs "cblas")
list(APPEND requiredlibs "lapack")

list(APPEND requiredlibs X11)
list(APPEND requiredlibs png jpeg)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${DLIB_PATH}/include)
link_directories(${DLIB_PATH}/lib)
link_directories(${DLIB_PATH}/lib64)

add_executable(img-dlib "img_dlib.cc")
target_link_libraries(img-dlib dlib)
target_link_libraries(img-dlib  ${requiredlibs})

