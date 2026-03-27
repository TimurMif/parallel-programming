cmake_minimum_required(VERSION 3.15)
project(Main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(MPI_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/msmpi.lib")

include_directories(${MPI_INCLUDE_PATH})

add_executable(main main.cpp)

target_link_libraries(main ${MPI_LIBRARY})

if(WIN32)
    target_link_libraries(main -static)
endif()
