INC     := -I$(CUDA_PATH)include  -I.
LIB     := -L$(CUDA_PATH)lib64
SDKINC  := -I$(CUDA_PATH)samples/common/inc

NVCCFLAGS := -O3 -arch=sm_20 --use_fast_math -c
LIBS    := -lcufft -lcudart -lm

all: cuda gmres3

cuda: cuda.cu
	nvcc cuda.cu $(SDKINC) $(INC) $(NVCCFLAGS) $(LIB) $(LIBS)


gmres3: 	global_params.F90 parameters.F90 io_arpack.F90 spec.F90 main.F90 timederiv.F90 convol.F90 fft-fftw.F90 init.F90
	gfortran -O3 -fno-math-errno -o ../bin/gmres3 global_params.F90 parameters.F90 io_arpack.F90 main.F90 timederiv.F90 spec.F90 convol.F90 fft-fftw.F90 init.F90 cuda.o -L$(LIBRARY_PATH) -lfftw3 -lm $(LIB) $(LIBS) -L~/ARPACK/ -larpack
