cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME hw2)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
add_executable(task1 task1/task1.cpp commons/feature_extraction.cpp commons/hog_visualization.cpp)
add_executable(task2 task2/task2.cpp commons/feature_extraction.cpp commons/hog_visualization.cpp task2/RandomForest.cpp)
add_executable(task3 task3/task3.cpp commons/feature_extraction.cpp commons/hog_visualization.cpp task2/RandomForest.cpp)

target_link_libraries(task1 ${OpenCV_LIBS})
target_link_libraries(task2 ${OpenCV_LIBS})
target_link_libraries(task3 ${OpenCV_LIBS})