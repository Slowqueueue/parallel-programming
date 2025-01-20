#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif

#define PrimeSize 100000

void RunTest(int N, int power, int numofSummands);
double Benchmark(int* action(int, int, int), int i, int* result, int length, int power, int numofSummands);

int* RunCuda(int n, int power, int numofSummands);
__global__ void SearchCuda(int* result, int n, int amount, int power, int numofSummands);
__device__ void IsPrimeCuda(int number, int* r);

int IsPrime(int number);
int* Search(int n, int power, int numofSummands);

long long* ReadPrimes(int n, long long* simple);
int StrToInt(char* chars, int length);

__device__ int Primes[PrimeSize];
long long* PrimesLocal;

int main(int argc, char** argv)
{
	int N, power, numofSummands;
	N = atoi(argv[1]);
	power = atoi(argv[2]);
	if (power <= 0) {
		printf("Power must be greater than 0");
		return 0;
	}
	numofSummands = atoi(argv[3]);
	if (numofSummands != 1 && numofSummands != 2 && numofSummands != 3) {
		printf("Number of summands must be 1, 2 or 3");
		return 0;
	}
	RunTest(N, power, numofSummands);
}

void RunTest(int N, int power, int numofSummands)
{
	long long* simple = (long long*)malloc(5000000 * sizeof(long long));
	PrimesLocal = ReadPrimes(5000000, simple);
	cudaMemcpyToSymbol(Primes, &PrimesLocal[0], PrimeSize * sizeof(int));

	int* (*SequentialFunc)(int, int, int ) = &Search;
	int* (*ParallelFunc)(int, int, int) = &RunCuda;

	int result[] = { 0,0,0,0 };
	int checkbuff[] = { 0 };
	double time = 0;
		time = Benchmark(SequentialFunc, N, result, 4, power, numofSummands);
		if (numofSummands == 3)
		printf("CPU:%lfs | %d=%d^%d + %d^%d + %d^%d \n", time, result[0], result[1], power, result[2], power, result[3], power);
		if (numofSummands == 2)
		printf("CPU:%lfs | %d=%d^%d + %d^%d \n", time, result[0], result[1], power, result[2], power);
		if (numofSummands == 1)
		printf("CPU:%lfs | %d=%d^%d \n", time, result[0], result[1], power);
		checkbuff[0] = result[0];
		time = Benchmark(ParallelFunc, N, result, 4, power, numofSummands);
		if (result[0] == checkbuff[0] - 1) result[0]++;
		if (numofSummands == 3)
		printf("GPU:%lfs | %d=%d^%d + %d^%d + %d^%d \n\n", time, result[0], result[1], power, result[2], power, result[3], power);
		if (numofSummands == 2)
		printf("GPU:%lfs | %d=%d^%d + %d^%d \n\n", time, result[0], result[1], power, result[2], power);
		if (numofSummands == 1)
		printf("GPU:%lfs | %d=%d^%d \n\n", time, result[0], result[1], power);
}

double Benchmark(int* action(int, int, int), int i, int* result, int length, int power, int numofSummands)
{
	double start, end;

	start = omp_get_wtime();

	int* ActionResult = action(i, power, numofSummands);
	for (int i = 0; i < length; i++)
		result[i] = ActionResult[i];

	end = omp_get_wtime();
	return end - start;
}

int* RunCuda(int n, int power, int numofSummands)
{
	int amount;
	if (numofSummands == 3)
	amount = (int)(ceil(pow(n, (double)1 / power)) * ceil(pow(n, (double)1 / power)) * ceil(pow(n, (double)1 / power)));
	if (numofSummands == 2)
	amount = (int)(ceil(pow(n, (double)1 / power)) * ceil(pow(n, (double)1 / power)));
	if (numofSummands == 1)
	amount = (int)(ceil(pow(n, (double)1 / power)));

	int* localResult = (int*)calloc(amount * 4, sizeof(int));

	int* result;
	cudaMalloc((void**)&result, sizeof(int) * amount * 4);
	cudaMemset((void*)result, 0, sizeof(int) * amount * 4);

	int blocks = ceil((double)amount / 1024);
	if (blocks == 0)
		blocks = 1;
	int threads = amount;
	if (threads > 1024)
		threads = 1024;

	SearchCuda KERNEL_ARGS2 (blocks, threads) (result, n, amount, power, numofSummands);

	cudaDeviceSynchronize();

	cudaMemcpy(localResult, result, sizeof(int) * amount * 4, cudaMemcpyDeviceToHost);

	cudaFree(result);

	int* max = (int*)malloc(4 * sizeof(int));
	max[0] = 0;
	max[1] = 0;
	max[2] = 0;
	max[3] = 0;
	for (int i = 0; i < amount * 4; i += 4)
		if (localResult[i] != 0 && localResult[i] > max[0])
		{
			max[0] = localResult[i];
			max[1] = localResult[i + 1];
			max[2] = localResult[i + 2];
			max[3] = localResult[i + 3];
		}

	free(localResult);

	return &max[0];
}

__global__ void SearchCuda(int* result, int n, int amount, int power, int numofSummands)
{
	int count = blockIdx.x * blockDim.x + threadIdx.x;
	double limA, limB, limC;
	if (count <= amount)
	{
		if (numofSummands == 3) {
			limA = powf(n, (float)1 / power);
			limB = powf(n, (float)1 / power);
			limC = powf(n, (float)1 / power);
		}
		if (numofSummands == 2) {
			limA = powf(n, (float)1 / power);
			limB = powf(n, (float)1 / power);
		}
		if (numofSummands == 1) {
			limA = powf(n, (float)1 / power);
		}
		int a, b, c;
		if (numofSummands == 3) {
			c = fmod((double)count, ceil(limC));
			b = (count - c) / ceil(limC);
			a = (int)(b / ceil(limB));
			b = fmod((double)b, ceil(limB));
		}
		if (numofSummands == 2) {
			b = fmod((double)count, ceil(limB));
			a = (count - b) / ceil(limB);
		}
		if (numofSummands == 1) {
			a = fmod((double)count, ceil(limA));
		}
		int number = 0;
	
		if (numofSummands == 3) number = pow(a, power) + pow(b, power) + pow(c, power);
		if (numofSummands == 2) number = pow(a, power) + pow(b, power);
		if (numofSummands == 1) number = pow(a, power);
		int isAprime;
		int isBprime;
		int isCprime;
		if (numofSummands == 3) {
			IsPrimeCuda(a, &isAprime);
			IsPrimeCuda(b, &isBprime);
			IsPrimeCuda(c, &isCprime);
		}
		if (numofSummands == 2) {
			IsPrimeCuda(a, &isAprime);
			IsPrimeCuda(b, &isBprime);
		}
		if (numofSummands == 1) {
			IsPrimeCuda(a, &isAprime);
		}
		if (numofSummands == 3) {
			if (!(a == b || a == c || b == c) && isAprime == 1 && isBprime == 1 && isCprime == 1 && number < n)
			{
				result[count * 4 + 0] = number;
				result[count * 4 + 1] = a;
				result[count * 4 + 2] = b;
				result[count * 4 + 3] = c;
			}
		}
		if (numofSummands == 2) {
			if (!(a == b) && isAprime == 1 && isBprime == 1 && number < n)
			{
				result[count * 4 + 0] = number;
				result[count * 4 + 1] = a;
				result[count * 4 + 2] = b;

			}
		}
		if (numofSummands == 1) {
			if (isAprime == 1 && number < n)
			{
				result[count * 4 + 0] = number;
				result[count * 4 + 1] = a;
			}
		}
	}
}

__device__ void IsPrimeCuda(int number, int* r)
{
	for (int i = 0; i < PrimeSize && Primes[i] <= number; i++)
		if (Primes[i] == number)
		{
			*r = 1;
			return;
		}
	*r = 0;
}

int* Search(int n, int power, int numofSummands)
{
	int* result = (int*)calloc(4, sizeof(int));

	double limA = pow(n, (double)1 / power);
	double limB = pow(n, (double)1 / power);
	double limC = pow(n, (double)1 / power);
	if (numofSummands == 1) {
		for (int a = 0; a < limA; a++)
			{
			int number = (int)(pow(a, power));
				if (IsPrime(a) && result[0] < number && number < n)
				{
					result[0] = number;
					result[1] = a;
				}
			}
	}

	if (numofSummands == 2) {
		for (int a = 0; a < limA; a++)
			for (int b = 0; b < limB; b++)
				{
					int number = (int)(pow(a, power) + pow(b, power));

					if (!(a == b) && IsPrime(a) && IsPrime(b) && result[0] < number && number < n)
					{
						result[0] = number;
						result[1] = a;
						result[2] = b;
					}
				}
	}

	if (numofSummands == 3) {
		for (int a = 0; a < limA; a++)
			for (int b = 0; b < limB; b++)
				for (int c = 0; c < limC; c++)
				{
					int number = (int)(pow(a, power) + pow(b, power) + pow(c, power));

					if (!(a == b || a == c || b == c) && IsPrime(a) && IsPrime(b) && IsPrime(c) && result[0] < number && number < n)
					{
						result[0] = number;
						result[1] = a;
						result[2] = b;
						result[3] = c;
					}
				}
	}
	return result;
}
int IsPrime(int number)
{
	for (int i = 0; i < PrimeSize && PrimesLocal[i] <= number; i++)
		if (PrimesLocal[i] == number)
			return 1;
	return 0;
}
long long* ReadPrimes(int n, long long* simple)
{
	long long* a = (long long*)malloc(n * sizeof(long long));
	int i, j;
	for (i = 0; i < n; i++) {
		a[i] = i;
	}
	a[1] = 0;

	for (i = 2; i < n; i++) {
		if (a[i] != 0) {
			for (j = i * 2; j < n; j += i) {
				a[j] = 0;
			}
		}
	}

	int count = 0;
	for (i = 0; i < n; i++) {
		if (a[i] != 0) {
			count++;
		}
	}

	j = 0;
	for (i = 0; i < n; i++) {
		if (a[i] != 0) {
			simple[j] = a[i];
			++j;
		}
	}
	free(a);
	return &simple[0];
}
int StrToInt(char* chars, int length)
{
	int a = 0;
	for (int i = length - 1; i >= 0; i--)
		a += ((int)(chars[i] - '0') * pow(10, length - i - 1));
	return a;
}