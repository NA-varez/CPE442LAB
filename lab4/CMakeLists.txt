cmake_minimum_required(VERSION 3.1.0)
project(main)
find_package(OpenCV REQUIRED)  # Find OpenCV library
find_package(Threads REQUIRED) # Find pthread library
include_directories( ${OpenCV_INCLUDE_DIRS} "inc/*.hpp")

# Sets 'LIBS' to a list of libraries
#set(LIBS OpenCV_LIBS Threads::Threads)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

# Add executable and link pthread library
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} Threads::Threads)
#target_link_libraries(main PRIVATE Threads::Threads)


#file(GLOB SOURCES "src/*.cpp")


#add_executable(YourExecutableName your_source_files.cpp)
#target_link_libraries(YourExecutableName PRIVATE Threads::Threads)