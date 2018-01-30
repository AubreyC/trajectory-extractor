cmake_minimum_required(VERSION 2.8.9)
project(daz)

# Include header
include_directories(include)

# Add sources
set(SOURCES
    src/daz.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(daz ${SOURCES})
