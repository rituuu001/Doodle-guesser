cmake_minimum_required(VERSION 3.20)

project(DoodleGuesser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
#set the path to your OpenCV build directory
set(OpenCV_DIR "C:/Users/Rames/opencv/build")
find_package(OpenCV REQUIRED)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(RAYLIB_INSTALL OFF CACHE BOOL "" FORCE)
# raylib
add_subdirectory(libraries/raylib)

# Executable
add_executable(DoodleGuesser
    main.cpp
)

# Eigen + OpenCV
target_include_directories(DoodleGuesser PRIVATE
    libraries/eigen
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(DoodleGuesser PRIVATE
    raylib
    ${OpenCV_LIBS}
)