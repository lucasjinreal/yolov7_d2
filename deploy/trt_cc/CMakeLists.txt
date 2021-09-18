cmake_minimum_required(VERSION 2.6)

project(yolox)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# # cuda
# include_directories(/data/cuda/cuda-10.2/cuda/include)
# link_directories(/data/cuda/cuda-10.2/cuda/lib64)
# # cudnn
# include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
# link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)
# tensorrt
include_directories($ENV{HOME}/TensorRT/include)
link_directories($ENV{HOME}/TensorRT/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(demo_yolox ${PROJECT_SOURCE_DIR}/demo_yolox.cc)
target_link_libraries(demo_yolox nvinfer)
target_link_libraries(demo_yolox cudart)
target_link_libraries(demo_yolox ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

