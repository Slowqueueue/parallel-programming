#ifndef __CUDACC__  
#define __CUDACC__
#endif
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <omp.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__device__ int d_count;

void constantSubstrs(vector<vector<int>>& pos, int* count, int prevCharPos = -1, int curChar = 0)
{
	if (curChar == pos.size()) { (*count)++; return; }
	for (auto p : pos[curChar])
	{
		if (prevCharPos < p)
			constantSubstrs(pos, count, p, curChar + 1);
	}
}

void savePosChar(string& str, vector<vector<int>>& pos, char c, int charPos)
{
	if (pos.empty()) pos = vector<vector<int>>(str.size());
	for (int i = 0; i < str.size(); i++)
	{
		if (c == str[i]) pos[i].push_back(charPos);
	}
}

void getNumberOfBlocksNThreads(long long elemsCount, int* blocks, int* threads)
{
	const int THREADS_PER_BLOCK = 64;
	*blocks = (elemsCount - 1) / THREADS_PER_BLOCK + 1;
	*threads = (elemsCount < THREADS_PER_BLOCK ? elemsCount : THREADS_PER_BLOCK);
}

__device__ bool isCombinationValid(int* pos, int* sizes, int arraySize, long long combination)
{
	int prevCharPos = -1;
	int sizesSum = 0;
	for (int i = 0; i < arraySize; i++)
	{
		long long p = combination % sizes[i];
		if (prevCharPos > pos[sizesSum + p]) return false;
		combination /= sizes[i];
		prevCharPos = pos[sizesSum + p];
		sizesSum += sizes[i];
	}
	return true;
}

__global__ void constantSubstrs(int* pos, long long combinations, int* sizes,
	int arraySize, int combinationsPerThread)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int blockCombinations;
	__shared__ int blockCombinationsSummed;
	if (id == 0) d_count = 0;
	if (threadIdx.x == 0)
	{
		blockCombinations = 0;
		blockCombinationsSummed = 0;
	}
	int threadCombinations = 0;
	long long startCombination = id * combinationsPerThread;
	if (startCombination >= combinations) return;
	for (long long i = startCombination;
		i < combinations && i < startCombination + combinationsPerThread;
		i++)
	{
		if (isCombinationValid(pos, sizes, arraySize, i))
			threadCombinations++;
	}
	atomicAdd(&blockCombinations, threadCombinations);
	__syncthreads();
	if (atomicExch(&blockCombinationsSummed, 1) == 0)
		atomicAdd(&d_count, blockCombinations);
}

int main(int argc, char** argv)
{
	if (argc < 3) { printf("Error. Wrong arguments. Arguments: string length, substring\n"); return -1; }
	int strsize = atoi(argv[1]);
	string substring = argv[2];
	if (strsize < substring.size()) { printf("Error. Substring is longer than string\n"); return -1; }
	double start, end, time1, time2;
	srand(time(0));
	string str;
	str.resize(strsize);
	for (auto& c : str) c = substring[rand() % substring.size()];

	start = omp_get_wtime();
	vector<vector<int>> pos;
	for (int i = 0; i < str.size(); i++)
		savePosChar(substring, pos, str[i], i);
	for (auto& p : pos) if (p.empty()) return 0;
	vector<int> sizes;
	for (auto& p : pos)
		sizes.push_back(p.size());
	int sizesSum = 0;
	long long combinations = 1;
	for (auto& p : pos)
	{
		sizesSum += p.size();
		combinations *= p.size();
	}
	int* d_pos;
	cudaMalloc(&d_pos, sizesSum * sizeof(int));
	int offset = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		cudaMemcpy(d_pos + offset, &pos[i].front(), pos[i].size() * sizeof(int), cudaMemcpyHostToDevice);
		offset += pos[i].size();
	}
	int* d_sizes;
	cudaMalloc(&d_sizes, pos.size() * sizeof(int));
	cudaMemcpy(d_sizes, &sizes.front(), pos.size() * sizeof(int), cudaMemcpyHostToDevice);
	int combinationsPerThread = 100;
	int maxBlocks = 4000;
	int blocks, threads;
	getNumberOfBlocksNThreads(combinations / combinationsPerThread + 1, &blocks, &threads);
	while (blocks > maxBlocks)
	{
		combinationsPerThread *= 10;
		getNumberOfBlocksNThreads(combinations / combinationsPerThread + 1, &blocks, &threads);
	}
	constantSubstrs KERNEL_ARGS2(blocks, threads) (d_pos, combinations, d_sizes, sizes.size(), combinationsPerThread);
	int res1 = 0;
	cudaMemcpyFromSymbol(&res1, d_count, sizeof(int));
	cudaFree(d_pos);
	cudaFree(d_sizes);
	end = omp_get_wtime();
	time1 = end - start;

	printf("Parallel time: %f sec\nParallel result: %d substrings\n\n", time1, res1);

	start = omp_get_wtime();
	vector<vector<int>> pos2;
	for (int i = 0; i < str.size(); i++)
		savePosChar(substring, pos2, str[i], i);
	for (auto& p : pos2) if (p.empty()) return 0;
	int res2 = 0;
	constantSubstrs(pos2, &res2);
	end = omp_get_wtime();
	time2 = end - start;

	printf("Not parallel time: %f sec\nNot parallel result: %d substrings\n\n", time2, res2);

	double diff = abs(time2 - time1);
	cout << (time2 > time1 ? "Cuda is faster " : "Cuda is slower ") << "with difference " << diff << endl;
	std::ofstream fout("original_string.txt");
	fout << str;
	return 0;
}
