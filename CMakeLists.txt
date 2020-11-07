cmake_minimum_required(VERSION 3.0)

project(data_loader)

add_library(data_loader SHARED data_loader.cpp)

install(TARGETS data_loader RUNTIME DESTINATION .)
