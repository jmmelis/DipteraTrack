cmake_minimum_required (VERSION 2.8)
project(make_model_dat_files)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations")

find_package(Armadillo REQUIRED)

add_executable(make_model_dat_files save_model_dat_file.cpp)
