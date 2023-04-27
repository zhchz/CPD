# 简介

这个项目是 Coherent Point Drift 的 C++ 版本，主要参考这篇论文 [Point Set Registration: Coherent Point Drift](https://arxiv.org/pdf/0905.2635.pdf)，而且代码的实现是基于原论文的 [MATLAB 代码](https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift) 的忠实转码。计算结果跟原论文的结果是对其的。

# 使用方法

本项目是在 Ubuntu 环境下运行的，依赖于 CMake 编译。如果需要在其他平台，需要自行配置环境。

## 依赖项

### OpenCV

项目依赖于 OpenCV，项目中使用的版本是 OpenCV 4.5.4，请先安装好 OpenCV （不同版本应该不会有问题），并在 `CMakeLists.txt` 文件中替换 OpenCV 的路径。

### Eigen

在特征值求解以及求解线性方程中用到了 Eigen 库作为底层计算库，版本为 3.4.0，请先安装好 Eigen （建议使用同一个版本）库，并在 `CMakeLists.txt` 文件中同样替换路径。

### Spectra

大尺寸的特征值问题使用 [Spectra](https://github.com/yixuan/spectra) 库求解，不过为了采用 Krylov Schur 方法和 MATLAB 源码对其，我使用了这个 [branch](https://github.com/dotnotlock/spectra/tree/krylovschur)，在这个基础上做了一些小的改动，方便集成和调用。代码已经放到 `3rd` 目录下。

### csv2

为了读入数据，用了 [csv2](https://github.com/p-ranav/csv2) 库，代码也已经放到 `3rd` 中了

### matplot++

[matplot++](https://github.com/alandefreitas/matplotplusplus) 是 C++ 下的可视化的库，非常好用，便利程度跟 Python 的 maplotlib 很接近。代码在 `3rd` 中。

## 使用

使用如下命令可以直接编译，并且运行例子，matplot++ 要求支持 C++17 标准，所有需要调用
```
# 编译
./build.sh 7

# 调用
./run.sh
```
结果可以在 `results` 路径下看到。

如果需要，你可以将源码集成到你的项目中使用。