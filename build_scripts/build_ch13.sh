START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 13
cd $START_DIR/Chapter13/flashlight
mkdir build
cd build/
cmake -DCMAKE_PREFIX_PATH=$LIBS_DIR ..
cmake --build . --target all
