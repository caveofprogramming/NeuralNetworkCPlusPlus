cmake_minimum_required(VERSION 3.10)

project("Neural Network"    VERSION 1.0
                            DESCRIPTION "Neural network in C++"
                            LANGUAGES CXX)

set(SOURCE_DIR src)

set(SRC_FILES   ${SOURCE_DIR}/main.cpp
                ${SOURCE_DIR}/network.cpp
                ${SOURCE_DIR}/imageloader.cpp
                ${SOURCE_DIR}/image.cpp
)

add_executable(neuralnetwork ${SRC_FILES})

set_property(TARGET neuralnetwork PROPERTY CXX_STANDARD 17)
set_property(TARGET neuralnetwork PROPERTY CXX_STANDARD_REQUIRED ON)




