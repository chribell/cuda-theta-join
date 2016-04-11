#include "../inc/parser.h"
#include "../inc/helper.cuh"
#include <errno.h>

__global__ void naive_approach_extension(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, LLONG* partialSums)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int blockID = gridDim.x * blockIdx.y + blockIdx.x;

	if (row < rSize && col < sSize)
	{
		if (R[row].a > S[col].a)
		{
			atomicAdd(&partialSums[blockID], S[col].x);
			//atomicAdd(&partial_sums[blockID], 1); //for count aggregate function
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

	LLONG* hostPartialSums;
	LLONG* devicePartialSums;

	// allocate memory for relations
	checkErrors(cudaMalloc((void**)&(deviceR), sizeof(Tuple) * rSize));
	checkErrors(cudaMalloc((void**)&(deviceS), sizeof(Tuple) * sSize));

	// copy relations to gpu
	checkErrors(cudaMemcpy(deviceR, R, sizeof(Tuple) * rSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceS, S, sizeof(Tuple) * sSize, cudaMemcpyHostToDevice));

	dim3 threadBlock(blockSide, blockSide);
	dim3 grid((rSize / threadBlock.x) + 1, (sSize / threadBlock.y) + 1);

	// allocate memory for partial_sums
	checkErrors(cudaMalloc((void**)&(devicePartialSums), sizeof(LLONG) * grid.x * grid.y));

	cudaEventRecord(start);
	naive_approach_extension <<<grid, threadBlock>>>(deviceR, rSize, deviceS, sSize, devicePartialSums);
	cudaEventRecord(stop);

	checkErrors(cudaPeekAtLastError());
	checkErrors(cudaDeviceSynchronize());


	hostPartialSums = (LLONG*)malloc(sizeof(LLONG) * grid.x * grid.y);
	checkErrors(cudaMemcpy(hostPartialSums, devicePartialSums, sizeof(LLONG) * grid.x * grid.y, cudaMemcpyDeviceToHost));


	LLONG sum = 0;
	for(int i = 0; i < grid.x * grid.y; ++i)
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

	printf("Naive Approach Extension\n");

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
