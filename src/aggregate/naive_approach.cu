#include "../inc/parser.h"
#include "../inc/helper.cuh"
#include <errno.h>

__global__ void naive_approach(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, LLONG* sum)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < rSize && col < sSize)
	{
		if (R[row].a > S[col].a)
		{
			atomicAdd(sum, S[col].x);
			//atomicAdd(sum, 1); //for count aggregate function
		}
	}

}


float callKernel(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, int blockSide)
{
	float millis = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Tuple* deviceR;
	Tuple* deviceS;

	LLONG* hostSum;
	LLONG* deviceSum;

	// allocate memory for relations
	checkErrors(cudaMalloc((void**)&(deviceR), sizeof(Tuple) * rSize));
	checkErrors(cudaMalloc((void**)&(deviceS), sizeof(Tuple) * sSize));

	// copy relations to gpu
	checkErrors(cudaMemcpy(deviceR, R, sizeof(Tuple) * rSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceS, S, sizeof(Tuple) * sSize, cudaMemcpyHostToDevice));

	// allocate memory for overall sum
	checkErrors(cudaMalloc((void**)&(deviceSum), sizeof(LLONG)))

	dim3 threadBlock(blockSide, blockSide);
	dim3 grid((rSize / threadBlock.x) + 1, (sSize / threadBlock.y) + 1);
	printf("Grid (%d,%d), threadBlock (%d,%d)\t", grid.x, grid.y, threadBlock.x, threadBlock.y);

	cudaEventRecord(start);
	naive_approach <<<grid, threadBlock>>>(deviceR, rSize, deviceS, sSize, deviceSum);
	cudaEventRecord(stop);

	checkErrors(cudaPeekAtLastError());
	checkErrors(cudaDeviceSynchronize());

	hostSum = (LLONG*)malloc(sizeof(LLONG));
	checkErrors(cudaMemcpy(hostSum, deviceSum, sizeof(LLONG), cudaMemcpyDeviceToHost));

	printf("Sum: %llu\n", *hostSum);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(deviceR);
	cudaFree(deviceS);
	cudaFree(deviceSum);
	cudaDeviceReset();

	free(hostSum);

	return millis;
}

int main(int argc, char** argv)
{
	char* rPath;
	LLONG rSize;

	char* sPath;
	LLONG sSize;

	int blockSide;
	int repeats;

	char* eptr;
	// read input arguments
	if (argc != 7)
	{
		printf("Not enough arguments\n---------------\n");
		printf("1st:\t R path\n");
		printf("2nd:\t |R| (R size)\n");
		printf("3rd:\t S path\n");
		printf("4th:\t |S| (S size)\n");
		printf("5th:\t Thread block side (max 32)\n");
		printf("6th:\t Number of repeats\n");
		return 1;
	}

	rPath = argv[1];
	sPath = argv[3];

	rSize = strtoll(argv[2], &eptr, 10);
	sSize = strtoll(argv[4], &eptr, 10);

	blockSide = strtol(argv[5], &eptr, 10);
	repeats = strtol(argv[6], &eptr, 10);

	if(rSize == 0 || sSize == 0 || blockSide == 0 || repeats == 0)
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

	printf("Naive Approach\n");

	// call kernel multiple times
	float time_aggregate = 0.0;
	for(int i = 0; i < repeats; ++i)
	{
		time_aggregate += callKernel(R, rSize, S, sSize, blockSide);
	}

	// calculate and print average time
	printf("Execution time: %f\n", (time_aggregate / (float) repeats));

	free(R);
	free(S);

	return 0;
}
