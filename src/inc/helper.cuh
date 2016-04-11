#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <helper_cuda.h>
//#include <helper_functions.h>


#define checkErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
static __device__ __inline__ uint32_t __smid()
{
	uint32_t smid;
	asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
	return smid;
}*/
