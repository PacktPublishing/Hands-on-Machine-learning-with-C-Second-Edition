START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


#Chapter 2
cd $START_DIR/Chapter02/csv/eigen
mkdir build
cd build/
cmake -DCSV_LIB_PATH=$LIBS_DIR/sources/fast-cpp-csv-parser/ -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/dlib
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/mlpack
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/flashlight
mkdir build
cd build/
cmake -DCSV_LIB_PATH=$LIBS_DIR/sources/fast-cpp-csv-parser/ -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/img/dlib/
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/img/opencv/
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/hdf5
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/json
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all
