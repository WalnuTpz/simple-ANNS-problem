cmake_minimum_required(VERSION 3.10)

project(ANNS_Problem CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling...")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(search src/search.cpp)

add_executable(build_knn src/build_knn.cpp)

if(OpenMP_FOUND)
    target_link_libraries(search PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(build_knn PRIVATE OpenMP::OpenMP_CXX)
endif()