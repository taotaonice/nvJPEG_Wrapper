cmake_minimum_required(VERSION 2.8)
project(nvJPEG_wrapper)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(pybind11)
find_package(OpenCV)

include_directories(
  /usr/local/cuda-10.1/include/
  ../
  ${OpenCV_INCLUDE_DIRS}
  )
link_directories(
  /usr/local/cuda-10.1/lib64/
  )


pybind11_add_module(libnvjpeg main.cpp)
target_link_libraries(libnvjpeg
  nvjpeg
  cudart
  ${OpenCV_LIBS}
  )
