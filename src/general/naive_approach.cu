#include "../inc/parser.h"
#include "../inc/helper.cuh"
#include <errno.h>
#include <math.h>

__global__ void naive_approach(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, Result* outputResults, int* outputCounter)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < rSize && col < sSize)
	{
		if (R[row].a > S[col].a)
		{
			outputResults[atomicAdd(outputCounter, 1)] = { R[row].a, S[col].a, S[col].x };
		}
	}
}

TimePair callKernel(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, int blockSide, int deviceMemorySize)
{
	float executionTime = 0.0;
	float millis = 0.0;

	cudaEvent_t start, stop;
	checkErrors(cudaEventCreate(&start));
	checkErrors(cudaEventCreate(&stop));

	cudaEvent_t* intermediatesStart;
	cudaEvent_t* intermediateStop;

	Tuple* deviceR;
	Tuple* deviceS;

	// allocate memory for relations
	checkErrors(cudaMalloc((void**)&(deviceR), sizeof(Tuple) * rSize));
	checkErrors(cudaMalloc((void**)&(deviceS), sizeof(Tuple) * sSize));

	// copy relations to gpu
	checkErrors(cudaMemcpy(deviceR, R, sizeof(Tuple) * rSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceS, S, sizeof(Tuple) * sSize, cudaMemcpyHostToDevice));

	Result* hostOutputResults;
	Result* deviceOutputResults;

	int* outputCounter;

	LLONG maxSpace = 0;
	LLONG md = deviceMemorySize * 1024LL * 1024LL * 1024LL;
	int t = 0;
	int side = 0;

	LLONG rIndex;
	LLONG sIndex;

	int rPortion;
	int sPortion;

	int tempOffset = 0;
	int offset = 0;
	int rectangles = 0;
	int count = 0;

	checkErrors(cudaMalloc((void**)&(outputCounter), sizeof(int)));

	dim3 threadBlock(blockSide, blockSide);
	dim3 grid((rSize / threadBlock.x) + 1, (sSize / threadBlock.y) + 1);

	maxSpace = rSize * sSize * sizeof(Result);

	t = (maxSpace / md) + 1;

	printf("Md: %d, ||O||max: %llu, t: %d\n", deviceMemorySize, (maxSpace / 1024LL / 1024LL / 1024LL),t);
	printf("Grid: %dx%d, Thread Block: %dx%d\n", grid.x, grid.y, threadBlock.x, threadBlock.y);

	printf("Max results per pass %llu / %d = %d\n", md, sizeof(Result), (int)(md / (LLONG)sizeof(Result)));

	// allocate memory for device result
	checkErrors(cudaMalloc((void**)&(deviceOutputResults), sizeof(Result) * (int)(md / (LLONG)sizeof(Result))));

	// allocate 10gb -> 894784853 host memory to store results from device (13gb -> 1163220310)
	hostOutputResults = (Result*) malloc(sizeof(Result) * 1163220310);

	checkErrors(cudaEventRecord(start, 0));

	if(t == 1) 		// call kernel once and finish
	{
		naive_approach <<<grid, threadBlock, 0>>>(deviceR, rSize, deviceS, sSize, deviceOutputResults, outputCounter);
		checkErrors(cudaEventRecord(stop, 0));
		checkErrors(cudaMemcpyAsync(&tempOffset, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, 0));
		checkErrors(cudaMemcpyAsync(hostOutputResults, deviceOutputResults, tempOffset * sizeof(Result), cudaMemcpyDeviceToHost, 0));
		checkErrors(cudaEventElapsedTime(&executionTime, start, stop));
		offset += tempOffset;
	}
	else // call kernel multiple times
	{

		side = (int)sqrt((double)((rSize * sSize) / t));

		if((rSize % side == 0) && (sSize % side == 0) || ((rSize / t <= sSize ) && (sSize <= rSize)))
		{
			rPortion = side;
			sPortion = side;
		}
		else
		{
			rPortion = (int) rSize / t;
			sPortion = (int) sSize;
		}
		rectangles = getActualNumberOfRectangles(rPortion, sPortion, rSize, sSize);

		printf("Number of overall rectangles: %d\n", rectangles);

		intermediatesStart = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * rectangles);
		intermediateStop  = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * rectangles);

		for(int i = 0; i < rectangles; i++)
		{
			checkErrors(cudaEventCreate(&intermediatesStart[i]));
			checkErrors(cudaEventCreate(&intermediateStop[i]));
		}

		int zero = 0;
		rIndex = 0;
		while(rIndex < rSize)
		{
			sIndex = 0;
			while(sIndex < sSize)
			{
				if(rPortion > rSize - rIndex) rPortion = rSize - rIndex;
				if(sPortion > sSize - sIndex) sPortion = sSize - sIndex;

				checkErrors(cudaEventRecord(intermediatesStart[count], 0));
				naive_approach <<<grid, threadBlock, 0>>>(deviceR + rIndex, rPortion, deviceS + sIndex, sPortion, deviceOutputResults, outputCounter);
				checkErrors(cudaEventRecord(intermediateStop[count], 0));

				checkErrors(cudaMemcpyAsync(&tempOffset, outputCounter, sizeof(int), cudaMemcpyDeviceToHost, 0));
				checkErrors(cudaMemcpyAsync(outputCounter, &zero, sizeof(int), cudaMemcpyHostToDevice, 0));
				checkErrors(cudaMemcpyAsync(count == 0 ? hostOutputResults : hostOutputResults + offset, deviceOutputResults, tempOffset * sizeof(Result), cudaMemcpyDeviceToHost, 0));

				offset += tempOffset;
				count++;
				sIndex += sPortion;
			}
			rIndex += rPortion;
		}

	}


	checkErrors(cudaEventRecord(stop, 0));

	checkErrors(cudaPeekAtLastError());
	checkErrors(cudaDeviceSynchronize());

	checkErrors(cudaEventSynchronize(stop));
	checkErrors(cudaEventElapsedTime(&millis, start, stop));

	for(int i = 0; i < rectangles; i++)
	{
		float temp = 0.0;
		checkErrors(cudaEventElapsedTime(&temp, intermediatesStart[i], intermediateStop[i]));
		executionTime += temp;
	}

	checkErrors(cudaEventDestroy(start));
	checkErrors(cudaEventDestroy(stop));

	for(int i = 0; i < rectangles; i++)
	{
		checkErrors(cudaEventDestroy(intermediatesStart[i]));
		checkErrors(cudaEventDestroy(intermediateStop[i]));
	}


	checkErrors(cudaFree(deviceR));
	checkErrors(cudaFree(deviceS));

	checkErrors(cudaFree(deviceOutputResults));
	checkErrors(cudaDeviceReset());

	/*int res = assertResultsGenralAoS(R, rSize, S, sSize, hostOutputResults, offset);
	if(res == 0)
		printf("Success!\n");
	else
		printf("Fail :/\n");*/

	free(hostOutputResults);

	return {executionTime, millis};
}



int main(int argc, char** argv)
{
	char* rPath;
	LLONG rSize;

	char* sPath;
	LLONG sSize;

	int blockSide;
	int repeats;
	int device_memory;

	char* eptr;
	// read input arguments
	if (argc != 8)
	{
		printf("Not enough arguments\n---------------\n");
		printf("1st:\t R path\n");
		printf("2nd:\t |R| (R size)\n");
		printf("3rd:\t S path\n");
		printf("4th:\t |S| (S size)\n");
		printf("5th:\t Thread block side (max 32)\n");
		printf("6th:\t Available Device Memory (in GB)\n");
		printf("7th:\t Number of repeats\n");
		return 1;
	}

	rPath = argv[1];
	sPath = argv[3];

	rSize = strtoll(argv[2], &eptr, 10);
	sSize = strtoll(argv[4], &eptr, 10);

	blockSide = strtol(argv[5], &eptr, 10);
	device_memory = strtol(argv[6], &eptr, 10);
	repeats = strtol(argv[7], &eptr, 10);

	if(rSize == 0 || sSize == 0 || blockSide == 0 || repeats == 0 || device_memory == 0)
	{
		printf("Wrong input arguments (error: %d)", errno);
		return 1;
	}

	// allocate memory
	Tuple* R;
	Tuple* S;

	R = (Tuple*)malloc(sizeof(Relation) * rSize);
	S = (Tuple*)malloc(sizeof(Relation) * sSize);

	readRelationAoS(rPath, R);
	readRelationAoS(sPath, S);

	printf("Naive Approach\n");

	// call kernel multiple times
	TimePair* pairs = (TimePair*)malloc(sizeof(TimePair) * repeats);
	float executionTimeAggregate = 0.0;
	float overallTimeAggregate = 0.0;

	for(int i = 0; i < repeats; ++i)
	{
		pairs[i] = callKernel(R, rSize, S, sSize, blockSide, device_memory);
		executionTimeAggregate += pairs[i].executionTime;
		overallTimeAggregate += pairs[i].overallTime;
	}

	// calculate and print average time
	printf("-----------------\n");
	printf("Execution time (GPU): %f\n", (executionTimeAggregate / (float) repeats));
	printf("Transfer & overhead time: %f\n", (overallTimeAggregate - executionTimeAggregate) / (float) repeats);
	printf("-----------------\n");
	printf("Overall Execution time: %f\n", (overallTimeAggregate / (float) repeats));

	free(R);
	free(S);
	free(pairs);

	return 0;
}
