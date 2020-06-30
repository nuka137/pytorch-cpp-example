# pytorch-cpp-example
PyTorch C++ API Example

## Install libtorch

* Without cuda

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

* With cuda

```bash
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.1.zip
unzip libtorch-shared-with-deps-1.5.1.zip
```

## Build

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

