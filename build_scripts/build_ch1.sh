START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


# Chapter 1
cd $START_DIR/Chapter01/dlib_samples/
mkdir build 
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter01/eigen_samples/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter01/xtensor_samples/
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter01/blaze_samples/
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter01/arrayfire_samples/
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR
