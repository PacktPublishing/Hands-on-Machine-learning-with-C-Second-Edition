#.rst:
# Findmlpack
# -------------
#
# Find mlpack
#
# Find the mlpack C++ library
#
# Using mlpack::
#
#   find_package(mlpack REQUIRED)
#   include_directories(${MLPACK_INCLUDE_DIRS})
#   add_executable(foo foo.cc)
#
# This module sets the following variables::
#
#   mlpack_FOUND - set to true if the library is found
#   MLPACK_INCLUDE_DIRS - list of required include directories
#   MLPACK_VERSION_MAJOR - major version number
#   MLPACK_VERSION_MINOR - minor version number
#   MLPACK_VERSION_PATCH - patch version number
#   MLPACK_VERSION_STRING - version number as a string (ex: "1.0.4")

include(FindPackageHandleStandardArgs)

message(INFO "Looking mlpack in ${CMAKE_PREFIX_PATH}")
find_path(MLPACK_INCLUDE_DIR
    NAMES mlpack/core.hpp mlpack/prereqs.hpp
    PATHS "${CMAKE_PREFIX_PATH}/include/"
)

find_package_handle_standard_args(mlpack
    REQUIRED_VARS MLPACK_INCLUDE_DIR
)

if(mlpack_FOUND)
  set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIR})
endif()

# Hide internal variables
mark_as_advanced(MLPACK_INCLUDE_DIR)