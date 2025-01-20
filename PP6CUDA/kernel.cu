#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
#include <string>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <vector>
#include "cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <omp.h>
using namespace std;

__device__ int device_offset;
__host__ __device__ bool FindNumberM(unsigned long long n, unsigned long long* respower = 0, unsigned long long* resnumber = 0);
__global__ void KernelFunction(unsigned long long elementsINthread, unsigned long long NumbertoFind);
unsigned long long FindCUDA(unsigned long long NumbertoFind);

int main(int argc, char* argv[]) {
	double start, end;
	unsigned long long result1, result2, number, power;
	vector<unsigned long long> sequence;
	if (argc != 2) {
		printf("Wrong args. N\n");
		return -1;
	}
	unsigned long long N = strtoll(argv[1], NULL, 10);
	if (N < 1) {
		printf("N must be greater than 0\n");
		return -1;
	}
	printf("N = %llu\n", N);
	start = omp_get_wtime();
	for (unsigned long long i = N + 1; ; i++) if (FindNumberM(i)) { result1 = i; break; }
	end = omp_get_wtime();
	printf("Sequential time: %fs\n", end - start);

	start = omp_get_wtime();
	result2 = FindCUDA(N + 1);
	end = omp_get_wtime();
	printf("Parallel time: %fs\n", end - start);

	cout << "Sequential and parallel results are " << (result1 == result2 ? "equal" : "different") << "\n";
	for (int i = 1; i < (int)log10((double)result2) + 1; i++) {
		unsigned long long split = pow(10, i);
		unsigned long long resblank = result2;
		while (resblank != 0) {
			if ((int)log10((double)resblank) + 1 < i) break;
			if ((int)log10((double)(resblank % split)) + 1 == i) sequence.push_back(resblank % split);
			resblank /= 10;
		}
	}
	cout << "Result: " << result2 << ": ";
	for (unsigned long long i = 0; i < sequence.size(); i++) cout << sequence[i] << (i == sequence.size() - 1 ? "" : "+");
	FindNumberM(result2, &power, &number);
	cout << "=" << (long long)pow(number, power) << "=" << number << "^" << power << endl;
	return 0;
}

__global__ void KernelFunction(unsigned long long elementsINthread, unsigned long long NumbertoFind) {
	unsigned long long ID = blockDim.x * blockIdx.x + threadIdx.x, startingpoint = NumbertoFind + ID * elementsINthread;
	for (unsigned long long i = startingpoint; i < startingpoint + elementsINthread; i++)
	{
		if (i - NumbertoFind > device_offset && device_offset != INT_MAX) return;
		if (FindNumberM(i)) { atomicMin(&device_offset, (int)(i - NumbertoFind)); return; }
	}
}

unsigned long long FindCUDA(unsigned long long NumbertoFind) {
	unsigned long long elementsINthread = 100, blocks = 64, threads = 64, offset = INT_MAX;
	cudaMemcpyToSymbol(device_offset, &offset, sizeof(int));
	while (true) {
		KernelFunction KERNEL_ARGS2(blocks, threads) (elementsINthread, NumbertoFind);
		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol(&offset, device_offset, sizeof(int));
		if (offset != INT_MAX) break;
		NumbertoFind += blocks * threads * elementsINthread;
	}
	return NumbertoFind + (unsigned long long)offset;
}

__host__ __device__ bool FindNumberM(unsigned long long n, unsigned long long* respower, unsigned long long* resnumber) {
	unsigned long long sumfin = 0, nbase = n;
	for (unsigned long long i = 1; i < (int)log10((double)n) + 1; i++) {
		unsigned long long split = pow(10, i), sumtemp = 0;
		while (n != 0) {
			if ((int)log10((double)n) + 1 < i) break;
			if ((int)log10((double)(n % split)) + 1 == i) sumtemp += n % split;
			n /= 10;
		}
		n = nbase;
		sumfin += sumtemp;
	}
	if (sumfin == 0 || sumfin == 1) return false;
	for (unsigned long long power = 2; ; power++) {
		double number = pow(sumfin, 1.0 / power);
		if ((int)number == number) {
			if (respower != nullptr) *respower = power;
			if (resnumber != nullptr) *resnumber = number;
			return true;
		}
		if (number < 2.0f && 1.0f < number) return false;
	}
}