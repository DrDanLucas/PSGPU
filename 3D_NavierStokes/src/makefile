INC     := -I$(CUDA_PATH)  -I.
LIB     := -L$(CUDA_PATH)lib64 
SDKINC  := -I$(CUDA_PATH)samples/common/inc

NVCCFLAGS := -O3 -arch=sm_20 --use_fast_math -c
LIBS    := -lcufft -lcudart -lm
  
all: cuda dns3 	

cuda:      cuda.cu kernels.h functions.h
	   nvcc cuda.cu $(SDKINC) $(INC) $(NVCCFLAGS) $(LIB) $(LIBS)

dns3: 	global_params.F90 parameters.F90 main.F90 init.F90 cuda.o
	mpif90 -O3 -fno-math-errno -mcmodel=large  -o ../bin/dns3 global_params.F90 parameters.F90 main.F90 init.F90 cuda.o $(LIB) $(LIBS)

rewrite: rewrite_plane

rewrite_plane:  global_params.F90 rewrite_plane.f 
	gfortran -O3 -fno-math-errno -mcmodel=large    -o ../bin/rewrite_plane global_params.F90 rewrite_plane.f 
