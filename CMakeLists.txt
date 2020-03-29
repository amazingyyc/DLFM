cmake_minimum_required(VERSION 3.5)

project(dlfm LANGUAGES CXX)

# c++17
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
endif(MSVC)

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# define for Eigen
add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DEIGEN_NO_CUDA)

# set for nnpack
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(-DHAS_NNPACK)
endif()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_subdirectory(third_party/NNPACK)
    include_directories(third_party/NNPACK/include)
endif()

include_directories(include)
include_directories(third_party)
include_directories(third_party/eigen)

set(DLFM_HEADER_FILES
  include/common/basic.h
  include/common/device.h
  include/common/element_type.h
  include/common/exception.h
  include/common/shape.h
  include/common/tensor_storage.h
  include/common/tensor.h

  include/math/fill.h
  include/math/relu.h
  include/math/sigmoid.h
  include/math/tanh.h
  include/math/unary_cwise.h
  include/math/binary_cwise.h
  include/math/cast.h
  include/math/transpose.h
  include/math/pad.h
  include/math/cat.h
  include/math/conv_transpose2d.h
  include/math/max_pooling2d.h
  include/math/upsample2d.h

  include/module/module.h
  include/module/conv2d.h
  include/module/conv_transpose2d.h
  include/module/sequential.h
  include/module/max_pooling2d.h
  include/module/relu.h
  include/module/unet.h
  include/module/tiny_unet.h
  include/module/sigmoid.h

  test/conv_transpose2d_test.h
  test/transpose_test.h
  test/cast_test.h
  test/cat_test.h
  test/pad_test.h
  test/conv2d_test.h
  test/max_pooling2d_test.h
  test/unet_test.h
  test/upsample2d_test.h
)

set(DLFM_SRC_FILES
  src/common/device.cc
  src/common/shape.cc
  src/common/tensor_storage.cc
  src/common/tensor.cc
  src/common/tensor_methods.cc

  src/math/fill.cc
  src/math/relu.cc
  src/math/sigmoid.cc
  src/math/tanh.cc
  src/math/unary_cwise.cc
  src/math/binary_cwise.cc
  src/math/cast.cc
  src/math/transpose.cc
  src/math/pad.cc
  src/math/cat.cc
  src/math/conv_transpose2d.cc
  src/math/max_pooling2d.cc
  src/math/upsample2d.cc

  src/module/module.cc
  src/module/conv2d.cc
  src/module/conv_transpose2d.cc
  src/module/sequential.cc
  src/module/max_pooling2d.cc
  src/module/relu.cc
  src/module/unet.cc
  src/module/tiny_unet.cc
  src/module/sigmoid.cc

  test/conv_transpose2d_test.cc
  test/cast_test.cc
  test/pad_test.cc
  test/cat_test.cc
  test/transpose_test.cc
  test/conv2d_test.cc
  test/max_pooling2d_test.cc
  test/unet_test.cc
  test/upsample2d_test.cc
  test/main.cc
 )

add_executable(dlfm ${DLFM_HEADER_FILES} ${DLFM_SRC_FILES})

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(dlfm PUBLIC pthreadpool nnpack)
endif()

# add_library(dlfm SHARED ${DLFM_HEADER_FILES} ${DLFM_SRC_FILES})