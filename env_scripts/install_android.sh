#!/usr/bin/env bash
set -x
set -e

START_DIR=$(pwd)
mkdir $START_DIR/android
cd $START_DIR/android

wget https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-android-sdk.zip
unzip opencv-4.10.0-android-sdk.zip

wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip
unzip commandlinetools-linux-9477386_latest.zip


# Set export JAVA_HOME variable to point where java SDK is installed  for example:
# export JAVA_HOME=~/android-studio/jbr

yes | ./cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_SDK_ROOT "cmdline-tools;latest"
yes | ./cmdline-tools/latest/bin/sdkmanager --licenses
yes | ./cmdline-tools/latest/bin/sdkmanager "platform-tools" "tools"
yes | ./cmdline-tools/latest/bin/sdkmanager "platforms;android-35"
yes | ./cmdline-tools/latest/bin/sdkmanager "build-tools;35.0.0"
yes | ./cmdline-tools/latest/bin/sdkmanager "system-images;android-35;google_apis;arm64-v8a"
yes | ./cmdline-tools/latest/bin/sdkmanager --install "ndk;26.1.10909125"

git clone https://github.com/pytorch/pytorch.git
cd pytorch/
git checkout v2.3.1
git submodule update --init --recursive

export ANDROID_NDK=$START_DIR/android/ndk/26.1.10909125
export ANDROID_ABI='arm64-v8a'
export ANDROID_STL_SHARED=1 

$START_DIR/android/pytorch/scripts/build_android.sh \
-DBUILD_SHARED_LIBS=ON \
-DUSE_VULKAN=OFF \
-DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
-DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)') \

# don't forget to upadte android project buils.gradle.kts file with next variables:
# val pytorchDir = "$START_DIR/android/pytorch/build_android/install/"
# val opencvDir = "$START_DIR/android/OpenCV-android-sdk/sdk/native/jni/"


