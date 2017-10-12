CXX=g++
nvcc=nvcc -c -arch=compute_20 -code=sm_20
#CXXFLAGS=-fopenmp -static -O3
#CXXFLAGS=-Xcompiler -fopenmp -fPIC -pipe -g -O3
##CXXFLAGS=-fopenmp -g -O3 -lgomp -lcuda -L/usr/local/cuda-6.0/targets/x86_64-linux/lib/ -lcudart -I /usr/local/cuda-6.0/targets/x86_64-linux/include/
CXXFLAGS=-fopenmp -g -O3 -lgomp -lcudart -L /usr/local/cuda-6.0/targets/x86_64-linux/lib/ -I /usr/local/cuda-6.0/targets/x86_64-linux/include/
#CXXFLAGS=-fopenmp -g -O3 -lgomp -lcudart -L /usr/local/cuda-6.0/lib/ -I /usr/local/cuda-6.0/include/#Use this row if the row above not work....
VERSION=1.4

#all: omp-pmf-train omp-pmf-predict
all: cuda-or-omp-pmf-train omp-pmf-predict

cuda-or-omp-pmf-train: CCDPP_onCUDA.o ALS_onCUDA.o ccd-r1.o util.o ccd-r1_original.o util_original.o ALS.o
	${CXX} ${CXXFLAGS} -o cuda-or-omp-pmf-train pmf-train.cpp ccd-r1.o util.o ccd-r1_original.o util_original.o CCDPP_onCUDA.o ALS_onCUDA.o ALS.o
	
	
omp-pmf-predict: pmf-predict.cpp pmf_original.h util_original.o
	${CXX} ${CXXFLAGS} -o omp-pmf-predict pmf-predict.cpp  util_original.o 	

ccd-r1.o: ccd-r1.cpp ccd-r1.o util.o
	${CXX} ${CXXFLAGS} -c -o ccd-r1.o ccd-r1.cpp
	
ALS.o: ALS.cpp ALS.o util.o
	${CXX} ${CXXFLAGS} -c -o ALS.o ALS.cpp		

util.o: util.h util.cpp
	${CXX} ${CXXFLAGS} -c -o util.o util.cpp
	
ccd-r1_original.o: ccd-r1_original.o util_original.o
	${CXX} ${CXXFLAGS} -c -o ccd-r1_original.o ccd-r1_original.cpp 

util_original.o: util_original.h util_original.cpp
	${CXX} ${CXXFLAGS} -c -o util_original.o util_original.cpp	

CCDPP_onCUDA.o: 
	${nvcc} CCDPP_onCUDA.cu

ALS_onCUDA.o: 
	${nvcc} ALS_onCUDA.cu	

tar: 
	make clean; cd ../;  tar cvzf libpmf-${VERSION}.tgz libpmf-${VERSION}/

clean:
	make -C python clean
	make -C R clean
	make -C matlab
	rm -rf  cuda-or-omp* *.o 

