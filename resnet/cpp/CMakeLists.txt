cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-cpp-example)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OPENCV_INCLUDE_DIRS}
)
add_executable(train model.cpp train.cpp)
add_executable(predict model.cpp predict.cpp)
target_link_libraries(train ${TORCH_LIBRARIES})
target_link_libraries(predict ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
set_property(TARGET train PROPERTY CXX_STANDARD 14)
set_property(TARGET predict PROPERTY CXX_STANDARD 14)

