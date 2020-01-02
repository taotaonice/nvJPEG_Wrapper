cmake_minimum_required(VERSION 2.5)
cmake_policy(SET CMP0003 NEW)
project(nvjpeg)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode;arch=compute_61,code=sm_61")

file(GLOB src "./main1.cpp")

find_package(OpenCV REQUIRED)
find_package(CUDA)

include_directories(
  /usr/local/cuda-10.1/include
  ${OpenCV_INCLUDE_DIRS}
  ./
  )

link_directories(
  /usr/local/cuda-10.1/lib64/
  )

add_executable(main ${src})

target_link_libraries(main 
  ${OpenCV_LIBS}
  nvjpeg
  cudart
  )
