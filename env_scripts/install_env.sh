#!/usr/bin/env bash
set -x
set -e

DEV_DIR=$(pwd)

mkdir -p libs
mkdir -p libs/sources

# Blaze
. ./install_lib.sh https://bitbucket.org/blaze-lib/blaze.git v3.8.2

# Firearray
. ./install_lib.sh https://github.com/arrayfire/arrayfire v3.8.3 -DBUILD_TESTS=OFF

# DLib
. ./install_lib.sh https://github.com/davisking/dlib v19.24

# Armadillo
. ./install_lib.sh https://gitlab.com/conradsnicta/armadillo-code 12.0.x

# xtl
. ./install_lib.sh https://github.com/xtensor-stack/xtl 0.7.5

# xtensor
. ./install_lib.sh https://github.com/xtensor-stack/xtensor 0.24.5

# xtensor-blas
. ./install_lib.sh https://github.com/xtensor-stack/xtensor-blas 0.20.0

# NlohmanJson
. ./install_lib.sh https://github.com/nlohmann/json.git v3.11.2 -DJSON_BuildTests=OFF

# mlpack
. ./install_lib.sh https://github.com/mlpack/mlpack 4.0.1 -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF -DDOWNLOAD_DEPENDENCIES=ON

# Eigen
. ./install_lib.sh https://gitlab.com/libeigen/eigen.git 3.4.0

# PyTorch
. ./install_lib.sh https://github.com/pytorch/pytorch v1.13.1 -DBUILD_PYTHON=OFF -DONNX_NAMESPACE=onnx_torch

# ONNX
. ./install_lib.sh https://github.com/onnx/onnx.git v1.13.1 -DONNX_NAMESPACE=onnx_torch

# HighFive
. ./install_lib.sh https://github.com/BlueBrain/HighFive v2.6.2

# cpp-httplib
. ./install_lib.sh https://github.com/yhirose/cpp-httplib v0.12.1

# PlotCpp
. ./checkout_lib.sh https://github.com/Kolkir/plotcpp c86bd4f5d9029986f0d5f368450d79f0dd32c7e4

# fast-cpp-csv-parser
. ./checkout_lib.sh https://github.com/ben-strasser/fast-cpp-csv-parser 4ade42d5f8c454c6c57b3dce9c51c6dd02182a66



# return back
cd $DEV_DIR


