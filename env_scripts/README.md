# Building development environment
There two main approaches to build development environment:
1. Configure local computer environment
2. Create a Docker image and container  

# Configure you GitHub account first
To be able to clone 3rd-party repositories you need a [GitHub](https://github.com) account. Then you will be able to configure GitHub authenticating with SSH as it is described in the article [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) this is the preffered way. Or using HTTPS and providing your username and password each time when a new repository will be cloned. Also, If you use 2FA to secure your GitHub account then you’ll need to use a personal access token instead of a password, as explained in the article [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

# Building development environment with Docker
1. Build Docker image.
```
cd env_scripts
docker build -t buildenv:1.0 .
```

2. Run a new container from the created image, and share code samples folder.
```
docker run -it -v [host_samples_path]:[container_samples_path] buildenv:1.0 bash
```

3. Samples from chapter 2 require accsess to your graphical environment to show images. You can share you X11 server with a Docker container. The following script shows how to run a container with graphics environment:
```
xhost +local:root
docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it -v [host_samples_path]:[container_samples_path] buildenv:1.0 bash
```

9. In the started container console session navigate to the `container_samples_path\build_scripts` folder. You will find there scripts to build samples for each chapter. The following script shows how to build samples for chapter 1:
```
./build_ch1.sh
```

# Building local development environment

1. Install Ubuntu 22.04

2. Run the following commands to configure the system:
```
apt-get install -y unzip
apt-get install -y build-essential
apt-get install -y gdb
apt-get install -y git
apt-get install -y cmake
apt-get install -y cmake-curses-gui
apt-get install -y python3
apt-get install -y python3-pip
apt-get install -y libblas-dev
apt-get install -y libopenblas-dev
apt-get install -y libfftw3-dev 
apt-get install -y libatlas-base-dev
apt-get install -y liblapacke-dev
apt-get install -y liblapack-dev
apt-get install -y libboost-all-dev
apt-get install -y libopencv-core4.5d
apt-get install -y libopencv-imgproc4.5d
apt-get install -q -y libopencv-dev
apt-get install -y libopencv-highgui4.5d
apt-get install -y libopencv-highgui-dev
apt-get install -y libhdf5-dev
apt-get install -y libjson-c-dev
apt-get install -y libx11-dev
apt-get install -y openjdk-8-jdk
apt-get install -y wget
apt-get install -y ninja-build
apt-get install -y gnuplot
apt-get install -y vim
apt-get install -y python3-venv

pip install pyyaml
pip install typing
pip install typing_extensions
```

3. Create build environment with the following commands \(We assume that the path "/path/to/examples/package/" contains extracted code samples package\):
```
cd ~/
mkdir development
cd ~/development
cp /path/to/examples/package/docker/checkout_lib.sh ~/development
cp /path/to/examples/package/docker/install_lib.sh ~/development
cp /path/to/examples/package/docker/install_env.sh ~/development
cp /path/to/examples/package/docker/install_android.sh ~/development
chmod 777 ~/development/checkout_lib.sh
chmod 777 ~/development/install_lib.sh
chmod 777 ~/development/install_env.sh
chmod 777 ~/development/install_android.sh
./install_env.sh
./android_env.sh
```

4. All third party libraries will be installed into the following directory:
```
$HOME/development/libs
```

5. Navigate to the `/path/to/examples/package/build_scripts` folder.

6. Choose the build script for the chapter you want to build, for example build script for the first chapter is `build_ch1.sh`

7. Updated the `LIBS_DIR` varibale in the script with the `$HOME/development/libs` value, or another one but it should the folder where all third party libraries are installed.

8. Run the build script to compile samples for the selected chapter.

# List of all third-party libraries
Name - tag - repository

Blaze - v3.8.2 - https://bitbucket.org/blaze-lib/blaze.git

Firearray - v3.8.3 - https://github.com/arrayfire/arrayfire

Armadillo - 12.0.x - https://gitlab.com/conradsnicta/armadillo-code

DLib - v19.24 - https://github.com/davisking/dlib

Eigen - 3.4.0 - https://gitlab.com/libeigen/eigen.git

mlpack - 4.0.1 - https://github.com/mlpack/mlpack

plotcpp - c86bd4f5d9029986f0d5f368450d79f0dd32c7e4 - https://github.com/Kolkir/plotcpp

PyTorch - v1.13.1 - https://github.com/pytorch/pytorch

xtensor - 0.24.5 - https://github.com/xtensor-stack/xtensor

xtensor-blas - 0.20.0 - https://github.com/xtensor-stack/xtensor-blas

xtl - 0.7.5 - https://github.com/xtensor-stack/xtl

OpenCV 4.5 - from the distribution installation package - https://github.com/opencv/opencv_contrib/releases/tag/4.5.0

fast-cpp-csv-parser - 4ade42d5f8c454c6c57b3dce9c51c6dd02182a66 - master - https://github.com/ben-strasser/fast-cpp-csv-parser

NlohmanJson - v3.11.2 - https://github.com/nlohmann/json.git

