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

file(GLOB_RECURSE DLFM_HEADER_FILES "include/*.h" "test/*.h")
file(GLOB_RECURSE DLFM_SRC_FILES "src/*.cc" "test/*.cc")

add_executable(dlfm ${DLFM_HEADER_FILES} ${DLFM_SRC_FILES})

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_libraries(dlfm PUBLIC pthreadpool nnpack)
endif()

# add_library(dlfm SHARED ${DLFM_HEADER_FILES} ${DLFM_SRC_FILES})