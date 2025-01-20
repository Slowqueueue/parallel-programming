#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int primes_size_local;
int* primes_local;
__device__ int primes_size;
__device__ int* primes;

int* SequentialMethod(int n);
int* ParallelMethod(int n);
__global__ void KernelFunc(int* result, int n, int amount);

int isprime(int number);
__device__ int isprimecuda(int number);

int* GetPrimeNumbers()
{
	FILE* file = fopen("primes.txt", "r");
	fpos_t position;
	fgetpos(file, &position);
	while (!feof(file))
		if (fgetc(file) == '\n') primes_size_local++;
	fsetpos(file, &position);
	int* primes = (int*)malloc(primes_size_local * sizeof(int));
	int ch;
	for (int i = 0; i < primes_size_local; i++)
	{
		char number[10];
		for (int n = 0; n < 10 && (ch = fgetc(file)) != EOF; n++)
		{
			number[n] = (char)ch;
			if (number[n] == '\n')
			{
				int a = 0;
				for (int j = n - 1; j >= 0; j--)
					a += ((int)(number[j] - '0') * pow(10, n - j - 1));
				primes[i] = a;
				break;
			}
		}
	}
	return primes;
}

int main(int argc, char* argv[])
{
	double start, end;
	int* result;
	int n = atoi(argv[1]);
	printf("N: %d\n", n);
	primes_local = GetPrimeNumbers();

	cudaMemcpyToSymbol(primes_size, &primes_size_local, sizeof(int));
	int* dynamic_area;
	cudaMalloc((void**)&dynamic_area, sizeof(int) * primes_size_local);
	cudaMemcpy(dynamic_area, primes_local, primes_size_local * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(primes, &dynamic_area, sizeof(int*));
	
	start = omp_get_wtime();
	result = SequentialMethod(n);
	end = omp_get_wtime();
	printf("Sequential: \nTime: %fs \nResult: %d = %d^2 + %d^3 + %d^4\n\n", end - start, result[0], result[1], result[2], result[3]);

	start = omp_get_wtime();
	result = ParallelMethod(n);
	end = omp_get_wtime();
	printf("Parallel: \nTime: %fs \nResult: %d = %d^2 + %d^3 + %d^4\n", end - start, result[0], result[1], result[2], result[3]);
}

int* SequentialMethod(int n)
{
	int* res = (int*)calloc(4, sizeof(int));
	double limitA = pow(n, (double)1 / 2);
	double limitB = pow(n, (double)1 / 3);
	double limitC = pow(n, (double)1 / 4);
	for (int a = 0; a < limitA; a++)
		for (int b = 0; b < limitB; b++)
			for (int c = 0; c < limitC; c++)
			{
				int number = (int)(pow(a, 2) + pow(b, 3) + pow(c, 4));
				if (isprime(a) && isprime(b) && isprime(c) && res[0] < number && number < n)
				{
					res[0] = number;
					res[1] = a;
					res[2] = b;
					res[3] = c;
				}
			}
	return res;
}

int* ParallelMethod(int n)
{
	int* res;
	int num = (int)(ceil(pow(n, (double)1 / 2)) * ceil(pow(n, (double)1 / 3)) * ceil(pow(n, (double)1 / 4)));
	cudaMalloc((void**)&res, sizeof(int) * num * 4);
	cudaMemset((void*)res, 0, sizeof(int) * num * 4);
	int* localRes = (int*)calloc(num * 4, sizeof(int));
	int blocks = ceil((double)num / 1024);
	if (blocks == 0) blocks = 1;
	int threads = num;
	if (threads > 1024) threads = 1024;

	KernelFunc KERNEL_ARGS2(blocks, threads) (res, n, num);

	cudaDeviceSynchronize();
	cudaMemcpy(localRes, res, sizeof(int) * num * 4, cudaMemcpyDeviceToHost);
	cudaFree(res);

	int* max = (int*)malloc(4 * sizeof(int));
	max[0] = 0, max[1] = 0, max[2] = 0, max[3] = 0;
	for (int i = 0; i < num * 4; i += 4)
		if (localRes[i] != 0 && localRes[i] > max[0])
		{
			max[0] = localRes[i];
			max[1] = localRes[i + 1];
			max[2] = localRes[i + 2];
			max[3] = localRes[i + 3];
		}
	free(localRes);
	return &max[0];
}

__global__ void KernelFunc(int* result, int n, int num)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id <= num)
	{
		double limitA = powf(n, (float)1 / 2);
		double limitB = powf(n, (float)1 / 3);
		double limitC = powf(n, (float)1 / 4);
		int a, b, c;

		c = fmod((double)id, ceil(limitC));
		b = (id - c) / ceil(limitC);
		a = (int)(b / ceil(limitB));
		b = fmod((double)b, ceil(limitB));

		int number = a * a + b * b * b + c * c * c * c;
		if (isprimecuda(a) && isprimecuda(b) && isprimecuda(c) && result[0] < number && number < n)
		{
			result[id * 4 + 0] = number;
			result[id * 4 + 1] = a;
			result[id * 4 + 2] = b;
			result[id * 4 + 3] = c;
		}
	}
}

int isprime(int number)
{
	for (int i = 0; i < primes_size_local && primes_local[i] <= number; i++)
		if (primes_local[i] == number) return 1;
	return 0;
}

__device__ int isprimecuda(int number)
{
	for (int i = 0; i < primes_size && primes[i] <= number; i++)
		if (primes[i] == number) return 1;
	return 0;
}