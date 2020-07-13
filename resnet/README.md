# ResNet

ResNet model with C++/Python language.


## C++


### 1. Download libtorch

Download libtorch (PyTorch library for C++), extract.


* Without CUDA (CPU)

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

* With CUDA (CPU+GPU)

```bash
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.1.zip
unzip libtorch-shared-with-deps-1.5.1.zip
```


### 2. Build

Build ResNet with cmake.

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```


### 3. Download MNIST dataset

Download MNIST dataset from [website](http://yann.lecun.com/exdb/mnist/) and locate to `mnist` directory.


### 4. Run

Run ResNet model.

```bash
./resnet
```


## Python


### 1. Install pytroch

Install pytorch package.


```bash
pip install pytorch
```


### 2. Run

Run ResNet model.  
MNIST data will be downloaded to `mnist` directory within the Python program.


```bash
python resnet.py
```

