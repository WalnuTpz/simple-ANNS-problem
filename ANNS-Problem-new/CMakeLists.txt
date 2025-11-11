cmake_minimum_required(VERSION 3.10)

project(ANNS_Problem CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

add_executable(search src/search.cpp)
add_executable(build_knn src/build_knn.cpp)
