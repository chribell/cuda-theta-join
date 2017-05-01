#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include "../inc/parser.h"

#define NUMTHREADS 6

pthread_t callThd[NUMTHREADS];
pthread_mutex_t mutexsum;


LLONG rSize, sSize;
Tuple* R = NULL;
Tuple* S = NULL;
long aggregate;

void* aggregateJoin(void* arg)
{
	long rPortion = rSize / NUMTHREADS;
	long offset = (long) arg;
	long start = offset * rPortion;
	long end = start + rPortion;
	long i, j;
	long sum = 0;
	
	// last thread gets extra work
	if (offset == NUMTHREADS - 1 && end < rSize)
	{
		end += rSize - end;
	}
	
	
	for (i = start; i < end; ++i)
	{
		for (j = 0; j < sSize; ++j)
		{
			if (R[i].a > S[j].a)
			{
				sum += S[j].x;
			}
		}
	}
   
   
   pthread_mutex_lock (&mutexsum);
   aggregate += sum;
   pthread_mutex_unlock (&mutexsum);
   pthread_exit((void*)0);
	
}


int main(int argc, char** argv)
{
	char* rPath;
	char* sPath;
	
	char* eptr;
	// read input arguments
	if (argc != 5)
	{
		printf("Not enough arguments\n---------------\n");
		printf("1st:\t R path\n");
		printf("2nd:\t |R| (R size)\n");
		printf("3rd:\t S path\n");
		printf("4th:\t |S| (S size)\n");
		return 1;
	}

	rPath = argv[1];
	sPath = argv[3];

	rSize = strtoll(argv[2], &eptr, 10);
	sSize = strtoll(argv[4], &eptr, 10);

	if(rSize == 0 || sSize == 0)
	{
		printf("Wrong input arguments (error: %d)", errno);
		return 1;
	}
	
	struct timespec start, finish;
	double elapsed;

	R = (Tuple*)malloc(sizeof(Tuple) * rSize);
	S = (Tuple*)malloc(sizeof(Tuple) * sSize);

	readRelationAoS(rPath, R);
	readRelationAoS(sPath, S);

	void* status;
	pthread_attr_t attr;

	clock_gettime(CLOCK_MONOTONIC, &start);
	pthread_mutex_init(&mutexsum, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
   
	for (int i = 0 ; i < NUMTHREADS ; i++) 
	{
		pthread_create(&callThd[i], &attr, aggregateJoin, (void*)i); 
	}

	pthread_attr_destroy(&attr);

	for (int i = 0 ; i < NUMTHREADS ; i++) 
	   pthread_join(callThd[i], &status);

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	
	printf("Time elapsed (multi-threading): %f\n", elapsed);
   
	pthread_mutex_destroy(&mutexsum);
	//pthread_exit(0);
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	LLONG sum = 0;
	for (int i = 0; i < rSize; i++)
	{
		for (int j = 0; j < sSize; j++)
		{
			if (R[i].a > S[j].a)
			{
				sum += S[j].x;
			}
		}
		
	}
	
	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	
	
	printf("Time elapsed (single-thread): %f\n", elapsed);
	
	printf ("Sum =  %llu \n", aggregate);
	//printf ("Nested loop =  %llu \n", sum);

	
	
	free (R);
	free (S);
	
	
	return 0;
}
