START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 3
cd $START_DIR/Chapter05/dlib
mkdir build
cd build/
cmake cmake -DCMAKE_PREFIX_PATH=/development/libs/ -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all

cd $START_DIR/Chapter03/mlpack
mkdir build
cd build/
cmake cmake -DCMAKE_PREFIX_PATH=/development/libs/ -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all
