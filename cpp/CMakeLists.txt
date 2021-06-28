cmake_minimum_required(VERSION 3.0)

project(native_bits)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_library(native_bits SHARED training_data_loader.cpp compression.cpp)

find_package(Threads REQUIRED)

target_link_libraries(native_bits Threads::Threads)

install(
  TARGETS native_bits
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
