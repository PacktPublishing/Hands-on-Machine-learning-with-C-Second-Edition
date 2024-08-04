START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


#Chapter 12
cd $START_DIR/Chapter12/dlib
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter12/mlpack
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR  .. 
cmake --build . --target all

cd $START_DIR/Chapter12/flashlight
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter12/pytorch
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR .. 
cmake --build . --target all

cd $START_DIR/Chapter12/onnxruntime
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR .. 
cmake --build . --target all

