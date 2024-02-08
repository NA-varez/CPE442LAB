cmake_minimum_required(VERSION 2.8)
project(main)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} "inc/*.hpp")
#file(GLOB SOURCES "src/*.cpp")
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

