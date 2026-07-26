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


add_executable(test_fc_gradcheck
    tests/test_fc_gradcheck.cpp
    src/layers/fc_layer.cpp
)

target_include_directories(test_fc_gradcheck PRIVATE
    libraries/eigen
    include
)

# Executable
add_executable(DoodleGuesser
    src/app_main.cpp
    gui/canvas.cpp
    gui/ui.cpp
    
    src/layers/relu_layer.cpp
    src/layers/maxpool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax_layer.cpp
    src/layers/conv_layer.cpp

    src/dataset.cpp
    src/optimizer.cpp
    src/network.cpp
    src/preprocessing.cpp

)


# Eigen + OpenCV
target_include_directories(DoodleGuesser PRIVATE
    include
    libraries/eigen
    ${OpenCV_INCLUDE_DIRS}
    include
)
add_custom_command(TARGET DoodleGuesser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${OpenCV_DIR}/x64/vc16/bin/opencv_world4120d.dll
    $<TARGET_FILE_DIR:DoodleGuesser>
)

# Link libraries
target_link_libraries(DoodleGuesser PRIVATE
    raylib
    ${OpenCV_LIBS}
)

add_executable(test_dataset
    tests/test_dataset.cpp
    src/dataset.cpp
)

target_include_directories(test_dataset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Test executables
add_executable(test_relu
    tests/test_relu.cpp
    src/layers/relu_layer.cpp
)

target_include_directories(test_relu PRIVATE
    libraries/eigen
    include
)
