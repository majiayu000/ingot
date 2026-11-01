cmake_minimum_required(VERSION 3.24)
project(ingot VERSION 0.1.0 LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR "This project currently requires Apple Silicon macOS")
endif()

enable_testing()
add_subdirectory(macos)
