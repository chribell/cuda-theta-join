#include "../inc/parser.h"
#include "../inc/helper.cuh"
#include <errno.h>
#include <math.h>

__global__ void shared_approach_soa(int* Ra, int rSize, int* Sa, int* Sx, int sSize, int portionSize, Result* outputResults, int* outputCounter)
{
	extern __shared__ int shared[];
	int* sA = shared; // S.a
	int* sX = (int*)&sA[portionSize]; // S.x

	int rIndex = blockIdx.x * blockDim.x + threadIdx.x;

	int threadID = threadIdx.x; //local thread id, i.e. within block

	int rA = Ra[rIndex]; // store R.a to registers
	int sIndex = 0;

	while (sIndex < sSize) // iterate portions of the second array
	{
		if (threadID < portionSize)
		{
			sA[threadID] = Sa[sIndex + threadID];
			sX[threadID] = Sx[sIndex + threadID];
		}
		__syncthreads();

		//for (int i = 0; i < portionSize; ++i)
		for (int i = 0; i < (sSize - sIndex > portionSize ? portionSize : sSize - sIndex); ++i)
		{
			if (rA > sA[i] && rIndex < rSize)
			{
				outputResults[atomicAdd(outputCounter, 1)] = { rA, sA[i], sX[i] };
			}
		}
		sIndex += portionSize;

		__syncthreads();
	}
}

TimePair callKernel(Relation* R, LLONG rSize, Relation* S, LLONG sSize, int blockSize, int portionSize, int deviceMemorySize)
{
	float executionTime = 0.0;
	float millis = 0.0;

	cudaEvent_t start, stop;
	checkErrors(cudaEventCreate(&start));
	checkErrors(cudaEventCreate(&stop));

	cudaEvent_t* intermediateStart;
	cudaEvent_t* intermediateStop;

	unsigned int* deviceRindex;
	int* deviceRa;
	int* deviceRx;

	unsigned int* deviceSindex;
	int* deviceSa;
	int* deviceSx;

	// allocate memory for R relation
	checkErrors(cudaMalloc((void**)&(deviceRindex), sizeof(unsigned int) * rSize));
	checkErrors(cudaMalloc((void**)&(deviceRa), sizeof(int) * rSize));
	checkErrors(cudaMalloc((void**)&(deviceRx), sizeof(int) * rSize));

	// allocate memory for S relation
	checkErrors(cudaMalloc((void**)&(deviceSindex), sizeof(unsigned int) * sSize));
	checkErrors(cudaMalloc((void**)&(deviceSa), sizeof(int) * sSize));
	checkErrors(cudaMalloc((void**)&(deviceSx), sizeof(int) * sSize));

	// copy relations to gpu
	checkErrors(cudaMemcpy(deviceRindex, R->index, sizeof(unsigned int) * rSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceRa, R->a, sizeof(int) * rSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceRx, R->x, sizeof(int) * rSize, cudaMemcpyHostToDevice));

	checkErrors(cudaMemcpy(deviceSindex, S->index, sizeof(unsigned int) * sSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceSa, S->a, sizeof(int) * sSize, cudaMemcpyHostToDevice));
	checkErrors(cudaMemcpy(deviceSx, S->x, sizeof(int) * sSize, cudaMemcpyHostToDevice));

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

	dim3 threadBlock(blockSize);
	dim3 grid((rSize / threadBlock.x) + 1);

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
		shared_approach_soa <<<grid, threadBlock, portionSize * sizeof(int) + portionSize * sizeof(int), 0>>>(deviceRa, rSize, deviceSa, deviceSx, sSize, portionSize, deviceOutputResults, outputCounter);
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

		intermediateStart = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * rectangles);
		intermediateStop  = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * rectangles);

		for(int i = 0; i < rectangles; i++)
		{
			checkErrors(cudaEventCreate(&intermediateStart[i]));
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

				checkErrors(cudaEventRecord(intermediateStart[count], 0));
				shared_approach_soa <<<grid, threadBlock, portionSize * sizeof(int) + portionSize * sizeof(int), 0>>>(deviceRa + rIndex, rPortion, deviceSa + sIndex, deviceSx + sIndex, sPortion, portionSize, deviceOutputResults, outputCounter);
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
		checkErrors(cudaEventElapsedTime(&temp, intermediateStart[i], intermediateStop[i]));
		executionTime += temp;
	}

	checkErrors(cudaEventDestroy(start));
	checkErrors(cudaEventDestroy(stop));

	for(int i = 0; i < rectangles; i++)
	{
		checkErrors(cudaEventDestroy(intermediateStart[i]));
		checkErrors(cudaEventDestroy(intermediateStop[i]));
	}


	checkErrors(cudaFree(deviceRindex));
	checkErrors(cudaFree(deviceRa));
	checkErrors(cudaFree(deviceRx));

	checkErrors(cudaFree(deviceSindex));
	checkErrors(cudaFree(deviceSa));
	checkErrors(cudaFree(deviceSx));

	checkErrors(cudaFree(deviceOutputResults));
	checkErrors(cudaDeviceReset());

/*
	int res = assertResultsGenralSoA(R, rSize, S, sSize, hostOutputResults, offset);
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

	int blockSize;
	int repeats;
	int device_memory;
	int portionSize;

	char* eptr;
	// read input arguments
	if (argc != 9)
	{
		printf("Not enough arguments\n---------------\n");
		printf("1st:\t R path\n");
		printf("2nd:\t |R| (R size)\n");
		printf("3rd:\t S path\n");
		printf("4th:\t |S| (S size)\n");
		printf("5th:\t Thread block size (max 1024)\n");
		printf("6th:\t Portion size (max 32)\n");
		printf("7th:\t Available Device Memory (in GB)\n");
		printf("8th:\t Number of repeats\n");
		return 1;
	}

	rPath = argv[1];
	sPath = argv[3];

	rSize = strtoll(argv[2], &eptr, 10);
	sSize = strtoll(argv[4], &eptr, 10);

	blockSize = strtol(argv[5], &eptr, 10);
	portionSize = strtol(argv[6], &eptr, 10);
	device_memory = strtol(argv[7], &eptr, 10);
	repeats = strtol(argv[8], &eptr, 10);

	if(rSize == 0 || sSize == 0 || blockSize == 0 || repeats == 0 || device_memory == 0 || portionSize == 0)
	{
		printf("Wrong input arguments (error: %d)", errno);
		return 1;
	}

	// allocate memory
	Relation* R;
	Relation* S;

	R = (Relation*)malloc(sizeof(Relation));
	S = (Relation*)malloc(sizeof(Relation));

	readRelationSoA(rPath, R, rSize);
	readRelationSoA(sPath, S, sSize);

	printf("Shared Memory Approach (SoA)\n");

	// call kernel multiple times
	TimePair* pairs = (TimePair*)malloc(sizeof(TimePair) * repeats);
	float executionTimeAggregate = 0.0;
	float overallTimeAggregate = 0.0;

	for(int i = 0; i < repeats; ++i)
	{
		pairs[i] = callKernel(R, rSize, S, sSize, blockSize, portionSize, device_memory);
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
