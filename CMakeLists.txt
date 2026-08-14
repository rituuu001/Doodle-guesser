cmake_minimum_required(VERSION 3.20)

project(DoodleGuesser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======================================================
# OpenCV
# ======================================================
find_package(OpenCV REQUIRED)

# ======================================================
# Raylib
# ======================================================

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(RAYLIB_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(libraries/raylib)

# ======================================================
# Common CNN Sources
# ======================================================

set(CNN_SOURCES
    src/layers/conv_layer.cpp
    src/layers/relu_layer.cpp
    src/layers/maxpool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax_layer.cpp

    src/network.cpp
    src/dataset.cpp
    src/loss.cpp
    src/serializer.cpp
)

# ======================================================
# CNN Library
# ======================================================

add_library(cnn STATIC ${CNN_SOURCES})

target_include_directories(cnn PUBLIC
    include
    libraries/eigen
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cnn PUBLIC
    ${OpenCV_LIBS}
)

# ======================================================
# TRAIN
# ======================================================

add_executable(train
    src/train_main.cpp
)

target_link_libraries(train PRIVATE
    cnn
)


# ======================================================
# PREDICT GUI
# ======================================================

add_executable(predict
    src/app_main.cpp
    src/predictor.cpp
    gui/canvas.cpp
    gui/ui.cpp
    src/preprocessing.cpp
)

target_link_libraries(predict PRIVATE
    cnn
    raylib
)

# ======================================================
# UNIT TESTS
# ======================================================

enable_testing()

add_executable(test_relu
    tests/test_relu.cpp
)

target_link_libraries(test_relu PRIVATE cnn)

add_executable(test_dataset
    tests/test_dataset.cpp
)

target_link_libraries(test_dataset PRIVATE cnn)

add_executable(test_fc_gradcheck
    tests/test_fc_gradcheck.cpp
)

target_link_libraries(test_fc_gradcheck PRIVATE cnn)

add_executable(test_grad_check
    tests/test_grad_check.cpp
)

target_link_libraries(test_grad_check PRIVATE cnn)

add_executable(test_overfit_sample
    tests/test_overfit_sample.cpp
)

target_link_libraries(test_overfit_sample PRIVATE cnn)

add_test(NAME ReLU COMMAND test_relu)
add_test(NAME Dataset COMMAND test_dataset)
add_test(NAME FCGrad COMMAND test_fc_gradcheck)
add_test(NAME GradCheck COMMAND test_grad_check)
add_test(NAME Overfit COMMAND test_overfit_sample)

# ======================================================
# RUN TARGETS
# ======================================================

add_custom_target(run_train
    COMMAND train
    DEPENDS train
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(run_evaluate
    COMMAND evaluate
    DEPENDS evaluate
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(run_predict
    COMMAND predict
    DEPENDS predict
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS
        test_relu
        test_dataset
        test_fc_gradcheck
        test_grad_check
        test_overfit_sample
)

# ======================================================
# Copy OpenCV DLL
# ======================================================

add_custom_command(TARGET predict POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120d.dll"
        $<TARGET_FILE_DIR:predict>
)