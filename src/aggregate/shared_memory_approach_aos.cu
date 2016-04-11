#include "../inc/parser.h"
#include "../inc/helper.cuh"
#include <errno.h>

template <int blockSize>
__global__ void shared_memory_approach_aos(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, LLONG* partialSums)
{
	extern __shared__ float sdata[];

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int blockID = gridDim.x * blockIdx.y + blockIdx.x;
	int threadID = (threadIdx.y * blockDim.x) + threadIdx.x; //local thread id, i.e. within block

	if(row > rSize) return;

	int rA = R[row].a;
	LLONG partialSum = 0;

	while (row < rSize && col < sSize)
	{
		if (rA > S[col].a)
		{
			partialSum += S[col].x;
		}
		col += blockDim.y;
	}

	sdata[threadID] = partialSum;
	__syncthreads();

	if ((blockSize >= 1024) && (threadID < 512))
	{
		sdata[threadID] = partialSum = partialSum + sdata[threadID + 512];
	}

	__syncthreads();

	if ((blockSize >= 512) && (threadID < 256))
	{
		sdata[threadID] = partialSum = partialSum + sdata[threadID + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) && (threadID < 128))
	{
		sdata[threadID] = partialSum = partialSum + sdata[threadID + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (threadID < 64))
	{
		sdata[threadID] = partialSum = partialSum + sdata[threadID + 64];
	}

	__syncthreads();

	if (threadID < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) partialSum += sdata[threadID + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			partialSum += __shfl_down(partialSum, offset);
		}
	}

	if (threadID == 0) partialSums[blockID] = partialSum;
}

float callKernel(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, dim3 threadBlock)
{
	float millis = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Tuple* deviceR;
	Tuple* deviceS;

	LLONG* hostPartialSums;
	LLONG* devicePartialSums;

	// allocate memory for relations
	checkErrors(cudaMalloc((void**)&(deviceR), sizeof(Tuple) * rSize));
	checkErrors(cudaMalloc((void**)&(deviceS), sizeof(Tuple) * sSize));

	// copy relations to gpu
	checkErrors(cudaMemcpy(deviceR, R, sizeof(Tuple) * rSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceS, S, sizeof(Tuple) * sSize, cudaMemcpyHostToDevice));

	dim3 grid((rSize / threadBlock.x) + 1);

	// allocate memory for partialSums
	checkErrors(cudaMalloc((void**)&(devicePartialSums), sizeof(LLONG) * grid.x));

	cudaEventRecord(start);
	shared_memory_approach_aos<1024> <<<grid, threadBlock, 1024 * sizeof(LLONG)>>>(deviceR, rSize, deviceS, sSize, devicePartialSums);
	cudaEventRecord(stop);

	checkErrors(cudaPeekAtLastError());
	checkErrors(cudaDeviceSynchronize());

	hostPartialSums = (LLONG*)malloc(sizeof(LLONG) * grid.x);
	checkErrors(cudaMemcpy(hostPartialSums, devicePartialSums, sizeof(LLONG) * grid.x, cudaMemcpyDeviceToHost));

	LLONG sum = 0;
	for(int i = 0; i < grid.x; ++i)
	{
		sum += hostPartialSums[i];
	}

	printf("Sum: %llu\n", sum);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(deviceR);
	cudaFree(deviceS);
	cudaFree(devicePartialSums);
	cudaDeviceReset();

	free(hostPartialSums);

	return millis;
}





int main(int argc, char** argv)
{
	char* rPath;
	LLONG rSize;

	char* sPath;
	LLONG sSize;

	int blockSideX;
	int blockSideY;
	int repeats;

	char* eptr;
	// read input arguments
	if (argc != 8)
	{
		printf("Not enough arguments\n---------------\n");
		printf("1st:\t R path\n");
		printf("2nd:\t |R| (R size)\n");
		printf("3rd:\t S path\n");
		printf("4th:\t |S| (S size)\n");
		printf("5th:\t Thread block side x \n");
		printf("6th:\t Thread block side y \n");
		printf("7th:\t Number of repeats\n");
		return 1;
	}

	rPath = argv[1];
	sPath = argv[3];

	rSize = strtoll(argv[2], &eptr, 10);
	sSize = strtoll(argv[4], &eptr, 10);

	blockSideX = strtol(argv[5], &eptr, 10);
	blockSideY = strtol(argv[6], &eptr, 10);
	repeats = strtol(argv[7], &eptr, 10);

	if(rSize == 0 || sSize == 0 || blockSideX == 0 || blockSideY == 0 || repeats == 0)
	{
		printf("Wrong input arguments (error: %d)", errno);
		return 1;
	}

	// allocate memory
	Tuple* R;
	Tuple* S;

	R = (Tuple*)malloc(sizeof(Tuple) * rSize);
	S = (Tuple*)malloc(sizeof(Tuple) * sSize);

	readRelationAoS(rPath, R);
	readRelationAoS(sPath, S);

	printf("Shared Memory Approach (AoS)\n");

	dim3 threadBlock(blockSideX, blockSideY);
	// call kernel multiple times
	float time_aggregate = 0.0;
	for(int i = 0; i < repeats; ++i)
	{
		time_aggregate += callKernel(R, rSize, S, sSize, threadBlock);
	}

	// calculate and print average time
	printf("Execution time: %f\n", (time_aggregate / (float) repeats));

	free(R);
	free(S);

	return 0;
}
