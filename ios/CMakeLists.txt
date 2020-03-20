# used for IOS build
cmake_minimum_required(VERSION 3.5)

# build for ios cmd:
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchain/iOS.cmake -DPLATFORM=OS64 -DENABLE_BITCODE=0 -DARCHS=arm64 -DDEPLOYMENT_TARGET=10
# Assertion failed: ((state_>>1) == 0), function ~Barrier, file third_party/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h, line 34.
project(dlfm_ios C CXX ASM)

# c++17
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# install folder
set(CMAKE_INSTALL_PREFIX ../dlfm)
SET(CMAKE_BUILD_TYPE "Release")

set(NNPACK_PREFIX ${PROJECT_SOURCE_DIR}/../third_party/NNPACK)
set(NNPACK_DEP_PREFIX ${PROJECT_SOURCE_DIR}/../deps)
set(DLFM_PREFIX ${PROJECT_SOURCE_DIR}/..)
set(THIRD_PARTY_PREFIX ${PROJECT_SOURCE_DIR}/../third_party)

# add nnpack
add_subdirectory(${NNPACK_PREFIX} NNPACK)

# src files
set(DLFM_SRCS
        ${DLFM_PREFIX}/src/common/device.cc
        ${DLFM_PREFIX}/src/common/shape.cc
        ${DLFM_PREFIX}/src/common/tensor_storage.cc
        ${DLFM_PREFIX}/src/common/tensor.cc
        ${DLFM_PREFIX}/src/common/tensor_methods.cc

        ${DLFM_PREFIX}/src/math/fill.cc
        ${DLFM_PREFIX}/src/math/relu.cc
        ${DLFM_PREFIX}/src/math/sigmoid.cc
        ${DLFM_PREFIX}/src/math/unary_cwise.cc
        ${DLFM_PREFIX}/src/math/cast.cc
        ${DLFM_PREFIX}/src/math/transpose.cc
        ${DLFM_PREFIX}/src/math/pad.cc
        ${DLFM_PREFIX}/src/math/cat.cc
        ${DLFM_PREFIX}/src/math/conv_transpose2d.cc
        ${DLFM_PREFIX}/src/math/max_pooling2d.cc

        ${DLFM_PREFIX}/src/module/module.cc
        ${DLFM_PREFIX}/src/module/conv2d.cc
        ${DLFM_PREFIX}/src/module/conv_transpose2d.cc
        ${DLFM_PREFIX}/src/module/sequential.cc
        ${DLFM_PREFIX}/src/module/max_pooling2d.cc
        ${DLFM_PREFIX}/src/module/relu.cc
        ${DLFM_PREFIX}/src/module/unet.cc
        ${DLFM_PREFIX}/src/module/tiny_unet.cc
        ${DLFM_PREFIX}/src/module/sigmoid.cc)

add_library(dlfm STATIC ${DLFM_SRCS})
target_include_directories(dlfm PRIVATE
        ${DLFM_PREFIX}/include
        # pthread
        ${NNPACK_DEP_PREFIX}/pthreadpool/include
        # nnpack
        ${NNPACK_PREFIX}/include
        # eigen
        ${THIRD_PARTY_PREFIX}
        ${THIRD_PARTY_PREFIX}/eigen)

target_compile_definitions(dlfm PUBLIC EIGEN_USE_THREADS=1 EIGEN_FAST_MATH=1 EIGEN_NO_CUDA=1 HAS_NNPACK=1)
# target_compile_definitions(dlfm PUBLIC HAS_NNPACK=1)
target_link_libraries(dlfm PUBLIC pthreadpool nnpack)

# .a file
install(TARGETS dlfm
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# head file
# INSTALL(FILES ${DLFM_INSTALL_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${DLFM_PREFIX}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${THIRD_PARTY_PREFIX}/eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})