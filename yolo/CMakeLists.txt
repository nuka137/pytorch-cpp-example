cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-cpp-example)

find_package(Torch REQUIRED)

add_executable(yolo yolo.cpp)
target_link_libraries(yolo "${TORCH_LIBRARIES}")
set_property(TARGET yolo PROPERTY CXX_STANDARD 14)

