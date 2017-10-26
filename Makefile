VERSION=0.1

CUDA_VERSION=6.5
# CUDA_VERSION=9.0

CUDA_ARCH=-arch=compute_20 -code=sm_20
# CUDA_ARCH=-arch=compute_35 -code=sm_35

LIBPATH=/usr/local/cuda-$(CUDA_VERSION)/lib64/
INCLUDEPATH=/usr/local/cuda-$(CUDA_VERSION)/include/

# WARNING_LVL=-Wall -Wextra
# CUDA_WARNING_LVL=-Xcompiler -Wall -Xcompiler -Wextra

CXX=g++
CXXFLAGS=$(WARNING_LVL) -fopenmp -g -O3 -lgomp -lcuda -lcudart -L /usr/local/cuda-$(CUDA_VERSION)/lib64/ -I /usr/local/cuda-$(CUDA_VERSION)/include/

nvcc=/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
nvccflags= $(CUDA_ARCH) $(CUDA_WARNING_LVL) -Xcompiler -fopenmp -Xcompiler -g -Xcompiler -O3 -lgomp -lcudart -L $(LIBPATH) -I $(INCLUDEPATH)

SOURCES = $(wildcard *.cpp)
CUDASOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cpp=.o)
CUDAOBJECTS = $(CUDASOURCES:.cu=.o)

all: cuda-or-omp-pmf-train omp-pmf-predict

cuda-or-omp-pmf-train: CCDPP_onCUDA.o ALS_onCUDA.o pmf-train.o ccd-r1.o util.o ccd-r1_original.o util_original.o ALS.o
	${CXX} $(CXXFLAGS) -o cuda-or-omp-pmf-train CCDPP_onCUDA.o ALS_onCUDA.o pmf-train.cpp ccd-r1.o util.o ccd-r1_original.o util_original.o ALS.o

omp-pmf-predict: pmf-predict.o util_original.o
	${CXX} $(CXXFLAGS) -o omp-pmf-predict pmf-predict.o util_original.o

%.o: %.cu
	${nvcc} $(nvccflags) -c $< -o $@

%.o: %.cpp
	${CXX} $(CXXFLAGS) -c $< -o $@

tar: 
	make clean; cd ../;  tar cvzf pmf_cuda-${VERSION}.tgz pmf_CUDA/

clean:
	rm -f $(OBJECTS) $(CUDAOBJECTS) cuda-or-omp-pmf-train omp-pmf-predict