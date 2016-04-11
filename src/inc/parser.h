#ifndef PARSER_H_
#define PARSER_H_

#include "definitions.h"
#include <string.h>

/*
	Tokenizer
*/
char** splitString(char* str, const char delimeter)
{
	char** tokens = NULL;
	char* token = NULL;
	char delim[1];
	delim[0] = delimeter;
	char* i = str;
	int count = 0;
	int index = 0;

	// count number of tokens --> number_of_delimeters + 1
	while (*i++) if (*i == delimeter) count++;
	count++;

	// allocate memory for tokens
	tokens = (char**)malloc(sizeof(char*) * count);

	if (tokens)
	{
		for (token = strtok(str, delim); token != NULL; token = strtok(NULL, delim))
		{
			*(tokens + index++) = strdup(token);
		}
	}

	return tokens;
}

/*
	Read input relation from disk (AoS data layout)
*/
void readRelationAoS(char* path, Tuple* tuples)
{
	FILE* fp = fopen(path, "r");
	char line[256];
	char** tokens;
	char* eptr;
	int index = 0;

	while (fgets(line, sizeof(line), fp))
	{
		tokens = splitString(line, ',');
		tuples[index].index  = strtol(tokens[0], &eptr, 10);
		tuples[index].a = strtol(tokens[1], &eptr, 10);
		tuples[index].x = strtol(tokens[2], &eptr, 10);
		index++;
	}

	fclose(fp);
}

/*
	Read input relation from disk (SoA data layout)
*/
void readRelationSoA(char* path, Relation* relation, LLONG numOfTuples)
{
	FILE* fp = fopen(path, "r");
	char line[256];
	char** tokens;
	char* eptr;
	int index = 0;

	relation->index = (unsigned int*)malloc(sizeof(unsigned int) * numOfTuples);
	relation->a = (int*)malloc(sizeof(int)* numOfTuples);
	relation->x = (int*)malloc(sizeof(int)* numOfTuples);

	while (fgets(line, sizeof(line), fp))
	{
		tokens = splitString(line, ',');
		relation->index[index]  = strtol(tokens[0], &eptr, 10);
		relation->a[index] = strtol(tokens[1], &eptr, 10);
		relation->x[index] = strtol(tokens[2], &eptr, 10);
		index++;
	}

	fclose(fp);
}

#endif // PARSER_H_
