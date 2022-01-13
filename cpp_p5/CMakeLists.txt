cmake_minimum_required(VERSION 3.16.13)
project(p5 VERSION 1.0)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} cnn.cpp MyMatrix.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})