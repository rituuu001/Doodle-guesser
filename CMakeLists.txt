cmake_minimum_required(VERSION 3.20)

project(DoodleGuesser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
#set the path to your OpenCV build directory
set(OpenCV_DIR "C:/Users/samirneupane/c++_sketch_detector/Doodle-guesser")
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
    src/layers/relu_layer.cpp
    src/layers/maxpool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax_layer.cpp
    src/layers/conv_layer.cpp
    src/dataset.cpp
    src/optimizer.cpp
    src/network.cpp
    src/loss.cpp

)


# Eigen + OpenCV
target_include_directories(DoodleGuesser PRIVATE
    include
    libraries/eigen
    ${OpenCV_INCLUDE_DIRS}
    include
)

# Link libraries
target_link_libraries(DoodleGuesser PRIVATE
    raylib
    ${OpenCV_LIBS}
)

add_executable(test_grad_check
    tests/test_grad_check.cpp
    src/layers/relu_layer.cpp
    src/layers/maxpool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax_layer.cpp
    src/layers/conv_layer.cpp
    src/network.cpp
    src/loss.cpp
)

target_include_directories(test_grad_check PRIVATE
    libraries/eigen
    include
)

add_executable(test_fc_gradcheck
    tests/test_fc_gradcheck.cpp
    src/layers/fc_layer.cpp
)

target_include_directories(test_fc_gradcheck PRIVATE
    libraries/eigen
    include
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

add_executable(test_overfit_sample
    tests/test_overfit_sample.cpp
    src/layers/relu_layer.cpp
    src/layers/maxpool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax_layer.cpp
    src/layers/conv_layer.cpp
    src/optimizer.cpp
    src/network.cpp
    src/loss.cpp
)

target_include_directories(test_overfit_sample PRIVATE
    libraries/eigen
    include
)

add_executable(train_main
    src/train_main.cpp
    src/layers/relu_layer.cpp
    src/layers/maxpool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax_layer.cpp
    src/layers/conv_layer.cpp
    src/dataset.cpp
    src/optimizer.cpp
    src/network.cpp
    src/loss.cpp
)

target_include_directories(train_main PRIVATE
    include
    libraries/eigen
)
# target_link_libraries(test_overfit_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
