cmake_minimum_required(VERSION 2.8.7)

project(Caffe C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake/Modules)

find_package(Protobuf REQUIRED)

set(Caffe_LINKER_LIBS "")

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/proto/*.proto")

PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

include_directories(${CMAKE_BINARY_DIR}) # for protobuf files

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include/caffe/proto/caffe.pb.h
  DEPENDS ${CMAKE_BINARY_DIR}/caffe.pb.h
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_BINARY_DIR}/caffe.pb.h ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include/caffe/proto/caffe.pb.h)

find_package(Boost 1.46 REQUIRED COMPONENTS system thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

find_package(OpenBlas REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${OpenBLAS_LIB})

find_package(Glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

find_package(GFlags REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

find_package(CUDA QUIET)
if(CUDA_FOUND)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
  add_definitions(-DCPU_ONLY)
endif(CUDA_FOUND)

## Other stuff

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include/)
include_directories(${PROJECT_BINARY_DIR}/../caffe/include/)

# for gtest
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/)

# for lmdb
add_definitions(-DUSE_LMDB)
find_package(LMDB REQUIRED)
include_directories(SYSTEM ${LMDB_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${LMDB_LIBRARIES})

## The files

file(GLOB Source GLOB "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/*.cpp")
file(GLOB SourceSolvers GLOB "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/solvers/*.cpp")
file(GLOB SourceUtil GLOB "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/util/*.cpp")
file(GLOB SourceGPUUtil GLOB "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/util/*.cu")
file(GLOB SourceLayers GLOB "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/layers/*.cpp")
file(GLOB SourceGPULayers GLOB "${CMAKE_CURRENT_SOURCE_DIR}/../caffe/src/caffe/layers/*.cu")

set(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"")

# force generation of proto header
add_library(proto STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include/caffe/proto/caffe.pb.h ${ProtoSources})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS})

if(CUDA_FOUND)
  cuda_add_library(ccaffe SHARED ${ProtoSources} ${ProtoHeaders} ${Source} ${SourceSolvers} ${SourceLayers} ${SourceGPULayers} ${SourceUtil} ${SourceGPUUtil} ccaffe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include/caffe/proto/caffe.pb.h)

  cuda_add_cublas_to_target(ccaffe)

  target_link_libraries(ccaffe ${PROTOBUF_LIBRARY} ${Caffe_LINKER_LIBS} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
else(CUDA_FOUND)
  add_library(ccaffe SHARED ${ProtoSources} ${ProtoHeaders} ${Source} ${SourceSolvers} ${SourceLayers} ${SourceUtil} ccaffe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../caffe/include/caffe/proto/caffe.pb.h)

  target_link_libraries(ccaffe ${PROTOBUF_LIBRARY} ${Caffe_LINKER_LIBS})
endif(CUDA_FOUND)
