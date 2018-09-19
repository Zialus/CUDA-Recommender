cmake_minimum_required(VERSION 3.9)
project(andre_cuda)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")

# Extra warning flags only during nvcc compilation
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall, -Wextra")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Weverything")
#endif ()

# Add debugging to CUDA NVCC flags for NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

set(EXTRA_NVCC_FLAGS -arch=sm_35 -Xptxas -dlcm=ca --std=c++11 -Xcompiler -rdynamic -lineinfo)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${EXTRA_NVCC_FLAGS})

# Extra warning flags during compilation
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
#endif ()

#set(WARNINGS "-Wno-format -Wno-unused-value -Wno-pointer-bool-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS}")

# Silence warning C4996 on Microsoft Compilers
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

file(GLOB src "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c")
file(GLOB hdr "${PROJECT_SOURCE_DIR}/src/*.hpp" "${PROJECT_SOURCE_DIR}/src/*.h")
file(GLOB cuda_cu "${PROJECT_SOURCE_DIR}/cuda_src/*.cu")
file(GLOB cuda_hdr "${PROJECT_SOURCE_DIR}/cuda_src/*.h")

include_directories("${PROJECT_SOURCE_DIR}/cuda_src/" "${PROJECT_SOURCE_DIR}/src/")
CUDA_ADD_EXECUTABLE(cuda_andre ${src} ${hdr} ${cuda_cu} ${cuda_hdr})
target_link_libraries(cuda_andre OpenMP::OpenMP_CXX)
