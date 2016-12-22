# Author      : Dongwei Wang
# wdw828@gmail.com
# version     : 1.0

BIN         		:= md5_gpu

# CUDA toolkit path
CUDA_INSTALL_PATH 	:= /usr/local/cuda

NVCC 				:= $(CUDA_INSTALL_PATH)/bin/nvcc
#
# include path
INCD 				:= -I$(CUDA_INSTALL_PATH)/include
# lib path
LIBS 				:= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart


# compile flags for g++ and nvcc
CXXFLAGS  			:= -O3
#NVCCFLAGS 			:= --ptxas-options=-v -O2 -arch=sm_20
NVCCFLAGS 			:= -Xptxas="-v" -O3 \
					-gencode arch=compute_50,code=sm_50 \
					# -gencode arch=compute_30,code=sm_30 \
					# -gencode arch=compute_35,code=sm_35 \
					# -gencode arch=compute_52,code=sm_52

# macro for files
CPP_SOURCES       	:= $(wildcard *.cpp)
CU_SOURCES        	:= md5_gpu.cu md5.cu
HEADERS           	:= $(wildcard *.h)
CPP_OBJS          	:= $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           	:= $(patsubst %.cu, %.o, $(CU_SOURCES))

# start to make
all: $(BIN)
$(BIN): $(CU_OBJS)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(CU_OBJS) $(LIBS)

md5_gpu.o: md5_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c md5_gpu.cu $(INCD)

md5.o: md5.cu
	$(NVCC) $(NVCCFLAGS) -c md5.cu $(INCD) # clean unnecessary files
clean:
	rm -f $(BIN) *.o tags

cppecho:
	echo "$(CPP_SOURCES)"

headerecho:
	echo "$(HEADERS)"

cuecho:
	echo "$(CU_SOURCES)"
