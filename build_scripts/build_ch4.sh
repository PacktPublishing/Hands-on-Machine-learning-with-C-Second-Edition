START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 4
cd $START_DIR/Chapter04/dlib
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all

cd $START_DIR/Chapter04/mlpack
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all
