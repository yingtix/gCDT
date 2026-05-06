cmake_minimum_required(VERSION 3.12)

project(MyProject LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_ARCHITECTURES "70" "75" "80" "86")

find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

file(GLOB SOURCES_CPP "*.cpp")

file(GLOB SOURCES_CU "*.cu")

set(SOURCES ${SOURCES_CPP} ${SOURCES_CU})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")