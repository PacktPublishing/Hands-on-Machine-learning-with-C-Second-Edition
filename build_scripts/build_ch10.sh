START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


#Chapter 10
cd $START_DIR/Chapter10/dlib
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter10/mlpack
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR  .. 
cmake --build . --target all

cd $START_DIR/Chapter10/flashlight
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter10/pytorch
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR .. 
cmake --build . --target all
