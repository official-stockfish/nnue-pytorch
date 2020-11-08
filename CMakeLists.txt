cmake_minimum_required(VERSION 3.0)

project(training_data_loader)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_library(training_data_loader SHARED training_data_loader.cpp)

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader Threads::Threads)

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
