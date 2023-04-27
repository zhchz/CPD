# Introduction

This project is a C++ implementation of Coherent Point Drift, mainly based on the paper [Point Set Registration: Coherent Point Drift](https://arxiv.org/pdf/0905.2635.pdf), and the code is a faithful transcription of the original [MATLAB code](https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift). The computed results are consistent with those in the original paper.

# Usage

This project runs on Ubuntu and requires CMake to build. If you need to run it on other platforms, you need to configure the environment yourself.

## Dependencies

### OpenCV

This project depends on OpenCV, and the version used in the project is OpenCV 4.5.4. Please install OpenCV (other versions should be compatible as well) and replace the path of OpenCV in the `CMakeLists.txt` file.

### Eigen

Eigen library is used as the underlying calculation library for eigenvalue calculation and linear equation solving, with version 3.4.0. Please install Eigen (it is recommended to use the same version) and replace the path in the `CMakeLists.txt` file.

### Spectra

[Spectra](https://github.com/yixuan/spectra) library is used to solve large-scale eigenvalue problems. However, in order to use the Krylov Schur method and following the original MATLAB source code, I used this [branch](https://github.com/dotnotlock/spectra/tree/krylovschur) and made some small modifications for easy integration and calling. The code has been placed in the `3rd` directory.

### csv2

To read data, [csv2](https://github.com/p-ranav/csv2) library is used, and the code has also been placed in the `3rd` directory.

### matplot++

[matplot++](https://github.com/alandefreitas/matplotplusplus) is a visualization library for C++, which is very easy to use and is similar to Python's matplotlib. The code is in the `3rd` directory.

## Usage

You can compile and run the example directly with the following command. matplot++ requires support for C++17 standard, so you need to call:
```
# Compile
./build.sh 7

# Run
./run.sh
```
The results can be found in the `results` directory.

You can integrate the source code into your own project for other using.