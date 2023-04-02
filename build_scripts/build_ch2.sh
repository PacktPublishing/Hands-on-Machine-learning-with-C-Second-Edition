START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


#Chapter 2
cd $START_DIR/Chapter02/csv/eigen
mkdir build
cd build/
cmake -DCSV_LIB_PATH=$LIBS_DIR/sources/fast-cpp-csv-parser/ -DEIGEN_LIB_PATH=$LIBS_DIR/include/eigen3/ ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/dlib
mkdir build
cd build/
cmake -DDLIB_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/img/dlib/
mkdir build
cd build/
cmake -DDLIB_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/img/opencv/
mkdir build
cd build/
cmake ..
cmake --build . --target all

