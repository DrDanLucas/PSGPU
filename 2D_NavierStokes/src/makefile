INC     := -I$(CUDA_PATH)/include -I.
LIB     := -L$(CUDA_PATH)/lib64
SDKINC  := -I$(CUDA_PATH)/samples/common/inc

NVCCFLAGS := -O3 -arch=sm_20 --ptxas-options=-v --use_fast_math -c
LIBS    := -lcufft -lcudart
 
all: cuda dns2 	

cuda:      cuda.cu
	   nvcc cuda.cu $(SDKINC) $(INC) $(NVCCFLAGS) $(LIB) $(LIBS)

dns2: 	global_params.F90 parameters.F90 main.F90 init.F90 cuda.o
	gfortran -O3 -mcmodel=medium -o ../bin/dns2 global_params.F90 parameters.F90 main.F90 init.F90 cuda.o $(LIB) $(LIBS)

rewrite: rewrite_vort rewrite_avgs rewrite_ZK

rewrite_vort:  global_params.F90 rewrite_vort.f 
	gfortran -O3 -mcmodel=large -o ../bin/rewrite_vort global_params.F90 rewrite_vort.f 

rewrite_avgs:  global_params.F90 rewrite_avgs.f 
	gfortran -O3 -mcmodel=large -o ../bin/rewrite_avgs global_params.F90 rewrite_avgs.f 

rewrite_ZK:  global_params.F90 rewrite_ZK.f 
	gfortran -O3 -mcmodel=large -o ../bin/rewrite_ZK global_params.F90 rewrite_ZK.f 
