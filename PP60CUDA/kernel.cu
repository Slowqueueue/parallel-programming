#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
using namespace std;

__device__ int d_offsetFromMin;
__host__ __device__ bool isNumberPrime(long long n);
__host__ __device__ bool isNumberCorrect(long long n);
__global__ void CudaFunction(long long min, int elemsPerThread, long long summands2[]);
long long NoCudaCalc(long long min, long long summands[]);
long long CudaCalc(long long min, long long summands2[]);
bool isEqual(long long res, long long res2, long long summands[], long long summands2[]);

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Wrong arguments: N\n");
		return -1;
	}
	long long N = atoi(argv[1]);
	long long N2 = N + N % 2 + 1;
	long long summands[3], summands2[3];
	long long res = 0, res2 = 0;
	double start, end;
	if (N < 0)
	{
		printf("N must not be negative\n");
		return -1;
	}
	printf("CPU...\n");

	start = omp_get_wtime();
	res = NoCudaCalc(N, summands);
	end = omp_get_wtime();

	printf("Calculated in %f sec\n", end - start);

	printf("GPU...\n");

	start = omp_get_wtime();
	res2 = CudaCalc(N2, summands2);
	end = omp_get_wtime();

	printf("Calculated in %f sec\n", end - start);

	if (isEqual(res, res2, summands, summands2) == 1)  printf("Numbers are equal\n");
	else printf("Numbers are different");

	printf("%lld = %lld + %lld + %lld\n", res, summands[0], summands[1], summands[2]);
	return 0;
}

long long NoCudaCalc(long long N, long long summands[])
{
	int k = 0;
	long long sum = 0;
	do {
		for (long long i = N; i < INFINITY; i++) {
			if (isNumberPrime(i) == true) {
				summands[k] = i;
				k++;
				if (k == 3) break;
			}
		}
		sum = summands[0] + summands[1] + summands[2];
		N = summands[0] + 1;
		k = 0;
	} while (isNumberPrime(sum) == false);
	return sum;
}

long long CudaCalc(long long N, long long summands2[])
{
	int elemsPerThread = 1;
	int blocks = 2, threads = 16;
	int resultOffset = INT_MAX;
	cudaMemcpyToSymbol(d_offsetFromMin, &resultOffset, sizeof(int));
	while (1)
	{
		CudaFunction << <blocks, threads >> > (N, elemsPerThread, summands2);
		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol(&resultOffset, d_offsetFromMin, sizeof(int));
		if (resultOffset != INT_MAX) break;
		N += blocks * threads * elemsPerThread * 2;
	}
	return N + (long long)resultOffset;
}

__global__ void CudaFunction(long long N, int elemsPerThread, long long summands2[])
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	long long i = N + id * 2;
	if (isNumberCorrect(i))
	{
		atomicMin(&d_offsetFromMin, (int)(i - N));
		return;
	}
}

bool isEqual(long long res, long long res2, long long summands[], long long summands2[]) {
	if (res == res2 && summands[0] == summands2[0] && summands[1] == summands2[1] && summands[2] == summands2[2]) return 0;
	return 1;
}

__host__ __device__ void Decompose(long long n, bool* isDecomposed, int currentCount = 0, long long currentSum = 0, long long previousPrime = 0)
{
	if (currentCount == 3)
	{
		if (currentSum == n) *isDecomposed = true;
		return;
	}
	else if (currentSum >= n) return;
	long long start = (previousPrime == 0 ? 2 : previousPrime + 2);
	if (previousPrime == 2) start = previousPrime + 1;
	int d = 1 + start % 2;
	if (currentCount == 2)
	{
		long long start = n - (n % 2 + 1);
		for (long long i = start; i >= n - currentSum && i > previousPrime; i -= 2)
		{
			if (*isDecomposed) return;
			if (isNumberPrime(i)) Decompose(n, isDecomposed, currentCount + 1, currentSum + i, i);
		}
		return;
	}
	for (long long i = start; i + currentSum <= n; i += d)
	{
		if (*isDecomposed) return;
		if (!isNumberPrime(i)) continue;
		if (3 % 2 != 0 && i == 2) continue;
		Decompose(n, isDecomposed, currentCount + 1, currentSum + i, i);
	}
}

__host__ __device__ bool isNumberPrime(long long n)
{
	if (n % 2 == 0 && n != 2) return false;
	for (long long i = 3; i * i <= n; i += 2)
		if (n % i == 0) return false;
	return true;
}

__host__ __device__ bool isNumberCorrect(long long n)
{
	if (!isNumberPrime(n)) return false;
	bool isDecomposed = false;
	Decompose(n, &isDecomposed);
	return isDecomposed;
}
