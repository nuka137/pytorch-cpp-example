cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-cpp-example)

find_package(Torch REQUIRED)

add_executable(resnet resnet.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}")
set_property(TARGET resnet PROPERTY CXX_STANDARD 14)

add_executable(summary summary.cpp)
target_link_libraries(summary "${TORCH_LIBRARIES}")
set_property(TARGET summary PROPERTY CXX_STANDARD 14)
