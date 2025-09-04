OpenCL_SDK=/global/homes/s/sgrauerg/NVIDIA_GPU_Computing_SDK
INCLUDE=-I${OpenCL_SDK}/OpenCL/common/inc -I${PATH_TO_UTILS}
LIBPATH=-L${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/targets/x86_64-linux/lib
LIB=-lOpenCL -lm
all:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${CFILES} ${LIB} -o ${EXECUTABLE}

clean:
	rm -f *~ *.exe *.txt
