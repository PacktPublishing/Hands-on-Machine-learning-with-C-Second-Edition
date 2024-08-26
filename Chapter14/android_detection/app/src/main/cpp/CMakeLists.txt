cmake_minimum_required(VERSION 3.22.1)
project("object-detection")

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 20)

# find_package can be used only if you have protobuf installed
# find_package(Torch REQUIRED)

set(Torch_DIR CACHE STRING "")
set(Torch_LIBS_DIR ${Torch_DIR}/lib/)
set(Torch_LIBS  ${Torch_DIR}/lib/libtorch_cpu.so ${Torch_DIR}/lib/libc10.so)
set(Torch_INCLUDE_DIRS ${Torch_DIR}/include)

set(OpenCV_DIR CACHE STRING "")
find_package(OpenCV REQUIRED)

add_compile_options(
        -Wall -Wextra
        $<$<CONFIG:RELEASE>:-Ofast>
        $<$<CONFIG:DEBUG>:-O0>
        $<$<CONFIG:DEBUG>:-ggdb3>
)

add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
)

include_directories(${ANDROID_NDK}/sources/android/native_app_glue
                    ${COMMON_SOURCE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Torch_INCLUDE_DIRS})

# build native_app_glue as a static lib
add_library(app_glue STATIC
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

set(SOURCES
    native-lib.cpp
    detector.cpp
    yolo.cpp
)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        m
        log
        app_glue
        camera2ndk
        mediandk
        opencv_java
        ${Torch_LIBS})