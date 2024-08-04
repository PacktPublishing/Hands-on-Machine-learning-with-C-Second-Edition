START_DIR=${PWD%/*}

# change these directories paths according to your configuration, also change them in the Chapter13/android_classify/local.proreties file
ANDROID_SDK_DIR=/development/android
ANDROID_PYTORCH_DIR=/development/android/pytorch/build_android/install

#Chapter 14 
# Export YOLOv5 model

# cd $START_DIR/Chapter14/
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3 export.py --weights yolov5s.torchscript --optimize
cp yolov5s.torchscript $START_DIR/Chapter14/android_detection/app/src/main/assets

cd $START_DIR/Chapter14/android_detection
mkdir app/src/main/jniLibs
mkdir app/src/main/jniLibs/arm64-v8a 
cp $ANDROID_PYTORCH_DIR/lib/libc10.so app/src/main/jniLibs/arm64-v8a/
cp $ANDROID_PYTORCH_DIR/lib/libtorch.so app/src/main/jniLibs/arm64-v8a/

. ./gradlew build

# Find resulting APK file in the Chapter14/android_classify/app/build/outputs/apk/release/ folder
# Notice that this script may fail if you run it into Docker container under Windows platform

# If you change build directories please update "pytorchDir" and "opencvDir variables in the build.gradle for the app
