cmake_minimum_required(VERSION 3.0)
project(andre_cuda)

find_package(CUDA QUIET)

set(CMAKE_CXX_STANDARD 11)

if (CUDA_FOUND)
message("Cuda was found!")
else()
	message("Cuda was NOT found!")
	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-6.5")
	message("CUDA_TOOLKIT_ROOT_DIR was set manually!")
endif (CUDA_FOUND)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")

set(CUDA_NVCC_FLAGS
  "-gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;")

# add -Wextra compiler flag for gcc compilations
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall,-Wextra")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -g")


# Add debugging to CUDA NVCC flags for NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")


file(GLOB src "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c" )
file(GLOB hdr "${PROJECT_SOURCE_DIR}/src/*.hpp" "${PROJECT_SOURCE_DIR}/src/*.h" )
file(GLOB cuda_cu "${PROJECT_SOURCE_DIR}/cuda_src/*.cu" )
file(GLOB cuda_hdr "${PROJECT_SOURCE_DIR}/cuda_src/*.h" )

set(predict_files "${PROJECT_SOURCE_DIR}/predict_src/pmf-predict.cpp"
	"${PROJECT_SOURCE_DIR}/src/pmf_original.h" "${PROJECT_SOURCE_DIR}/src/util_original.cpp")

add_executable(omp-pmf-predict ${predict_files})

include_directories("${PROJECT_SOURCE_DIR}/cuda_src/")
CUDA_ADD_EXECUTABLE(cuda-or-omp-pmf-train ${src} ${hdr} ${cuda_cu} ${cuda_hdr})

target_link_libraries(cuda-or-omp-pmf-train gomp)


