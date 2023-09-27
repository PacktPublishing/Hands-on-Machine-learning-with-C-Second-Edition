START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 7
cd $START_DIR/Chapter07/dlib
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all

cd $START_DIR/Chapter07/mlpack
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all

cd $START_DIR/Chapter07/flashlight
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ -DCSV_LIB_PATH=$LIBS_DIR/sources/fast-cpp-csv-parser/ ..
cmake --build . --target all
