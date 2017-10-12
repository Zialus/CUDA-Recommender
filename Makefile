VERSION=0.1

CUDA_VERSION=6.5

CXX=g++
CXXFLAGS=-fopenmp -g -O3 -lgomp -lcuda -lcudart -L /usr/local/cuda-$(CUDA_VERSION)/targets/x86_64-linux/lib/ -I /usr/local/cuda-$(CUDA_VERSION)/targets/x86_64-linux/include/

nvcc=/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
nvccflags=-c -arch=compute_20 -code=sm_20 -Xcompiler -Wextra -Xcompiler -fopenmp -Xcompiler -g -Xcompiler -O3 -lgomp -lcudart -L /usr/local/cuda-$(CUDA_VERSION)/targets/x86_64-linux/lib/ -I /usr/local/cuda-$(CUDA_VERSION)/targets/x86_64-linux/include/

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

#CXXFLAGS=-fopenmp -static -O3
#CXXFLAGS=-Xcompiler -fopenmp -fPIC -pipe -g -O3

all: cuda-or-omp-pmf-train omp-pmf-predict

cuda-or-omp-pmf-train: CCDPP_onCUDA.o ALS_onCUDA.o pmf-train.o ccd-r1.o util.o ccd-r1_original.o util_original.o ALS.o
	${CXX} $(CXXFLAGS) -o cuda-or-omp-pmf-train CCDPP_onCUDA.o ALS_onCUDA.o pmf-train.cpp ccd-r1.o util.o ccd-r1_original.o util_original.o ALS.o

omp-pmf-predict: pmf-predict.o util_original.o
	${CXX} $(CXXFLAGS) -o omp-pmf-predict pmf-predict.o util_original.o

%.o: %.cpp
	${CXX} $(CXXFLAGS) -c $< -o $@

CCDPP_onCUDA.o: CCDPP_onCUDA.cu
	${nvcc} $(nvccflags) CCDPP_onCUDA.cu

ALS_onCUDA.o: ALS_onCUDA.cu
	${nvcc} $(nvccflags) ALS_onCUDA.cu

tar: 
	make clean; cd ../;  tar cvzf pmf_cuda-${VERSION}.tgz pmf_CUDA/

clean:
	rm -f $(OBJECTS) cuda-or-omp-pmf-train omp-pmf-predict