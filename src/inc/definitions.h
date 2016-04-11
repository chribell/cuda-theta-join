#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#include <stdlib.h>
#include <stdio.h>

#define LLONG unsigned long long int

typedef struct {
	unsigned int index;
	int a;
	int x;
	char padding[4];
} Tuple;

typedef struct {
	unsigned int* index;
	int* a;
	int* x;
} Relation;

typedef struct {
	int rA;
	int sA;
	int sX;
} Result;

typedef struct {
	float executionTime;
	float overallTime;
} TimePair;

int getActualNumberOfRectangles(int rPortion, int sPortion, LLONG rSize, LLONG sSize)
{
	int rectangles = 0;

	LLONG rIndex = 0;
	LLONG sIndex = 0;

	rIndex = 0;
	while(rIndex < rSize)
	{
		sIndex = 0;
		while(sIndex < sSize)
		{
			if(rPortion > rSize - rIndex) rPortion = rSize - rIndex;
			if(sPortion > sSize - sIndex) sPortion = sSize - sIndex;
			rectangles++;
			sIndex += sPortion;
		}
		rIndex += rPortion;
	}
	return rectangles;
}

int assertResultsGenralAoS(Tuple* R, LLONG rSize, Tuple* S, LLONG sSize, Result* results, LLONG res_count)
{
	LLONG res_rA, res_sA, res_sX;
	res_rA = res_sA = res_sX = 0;

	for (LLONG i = 0; i < res_count; ++i)
	{
		res_rA += results[i].rA;
		res_sA += results[i].sA;
		res_sX += results[i].sX;
	}

	LLONG count, rA, sA, sX;
	count = rA = sA = sX = 0;
	for(int i = 0; i < rSize; i++)
		for(int j = 0; j < sSize; j++)
			if(R[i].a > S[j].a)
			{
				count++;
				rA += R[i].a;
				sA += S[j].a;
				sX += S[j].x;
			}

	return res_count == count && res_rA == rA && res_sA == sA && res_sX == sX ? 0 : 1;
}


int assertResultsGeneralSoA(Relation* R, LLONG rSize, Relation* S, LLONG sSize, Result* results, LLONG res_count)
{
	LLONG res_rA, res_sA, res_sX;
	res_rA = res_sA = res_sX = 0;

	for (LLONG i = 0; i < res_count; ++i)
	{
		res_rA += results[i].rA;
		res_sA += results[i].sA;
		res_sX += results[i].sX;
	}

	LLONG count, rA, sA, sX;
	count = rA = sA = sX = 0;
	for(int i = 0; i < rSize; i++)
		for(int j = 0; j < sSize; j++)
			if(R->a[i] > S->a[j])
			{
				count++;
				rA += R->a[i];
				sA += S->a[j];
				sX += S->x[j];
			}

	return res_count == count && res_rA == rA && res_sA == sA && res_sX == sX ? 0 : 1;
}


#endif // DEFINITIONS_H_
