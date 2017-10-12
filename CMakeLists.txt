cmake_minimum_required(VERSION 3.0)
project(andre_cuda)

find_package(CUDA QUIET)

if (CUDA_FOUND)
message("Cuda was found!")
else()
	message("Cuda was NOT found!")
	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-6.5")
	message("CUDA_TOOLKIT_ROOT_DIR was set manually!")
endif (CUDA_FOUND)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

# compared to class settings, we let NVidia's FindCUDA CMake detect 
# whether to build x64.  We tell it to support most devices, though, 
# to make sure more people can easily run class code without knowing 
# about this compiler argument
set(CUDA_NVCC_FLAGS
  "-gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_20,code=sm_20;")

# add -Wextra compiler flag for gcc compilations
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")


# add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
# set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

file(GLOB hdr *.hpp *.h )
file(GLOB cu  *.cu )

set(cuda-train-files ccd-r1_original.cpp util_original.cpp util.cpp ALS.cpp ccd-r1.cpp pmf-train.cpp)
set(predict_files pmf-predict.cpp pmf_original.h util_original.cpp)

CUDA_ADD_EXECUTABLE(cuda-or-omp-pmf-train ${cuda-train-files} ${hdr} ${cu})

add_executable(omp-pmf-predict ${predict_files})

target_link_libraries(cuda-or-omp-pmf-train gomp)
