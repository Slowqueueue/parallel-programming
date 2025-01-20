#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include <omp.h>
#include "cuda_runtime.h"


//Macro for kernel declarations(e0029fix)
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


double* sequantial(long long width, long long height, double temperature)
{
	long long j, i;
	double* matrixseq = (double*)malloc(sizeof(double) * width * height);
	double sequantialStart = omp_get_wtime();
	double iter_height = (temperature) / (height - 1);
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			matrixseq[i * width + j] = (temperature - iter_height * i) / (width - 1) * j;
	double sequantialEnd = omp_get_wtime();
	printf("Sequantial time is: %f seconds\n", sequantialEnd - sequantialStart);
	return(matrixseq);
}


__global__ void kernel(double* matrixkern, long long width, long long height, double temperature)
{
	long long i = blockIdx.x + threadIdx.x;
	long long j = blockIdx.y + threadIdx.y;
	if (i >= height || j >= width) return;
	double iter_height = (temperature) / (height - 1);
	matrixkern[i * width + j] = (temperature - iter_height * i) / (width - 1) * j;
}


double* parallel(long long width, long long height, double temperature)
{
	double* matrixbuff;
	double* matrixpar = (double*)calloc(height * width, sizeof(double));

	cudaMalloc((void**)&matrixbuff, sizeof(double) * width * height);
	cudaMemcpy(matrixbuff, matrixpar, width * height * sizeof(double), cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);
	dim3 gridSize(width, height);

	double parallelStart = omp_get_wtime();
	kernel KERNEL_ARGS2(gridSize, blockSize) (matrixbuff, width, height, temperature);
	double parallelEnd = omp_get_wtime();
	printf("Parallel time is: %f seconds\n", parallelEnd - parallelStart);

	cudaMemcpy(matrixpar, matrixbuff, width * height * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(matrixbuff);
	return (matrixpar);
}


int main(int argc, char** argv)
{
	double temperature;
	long long width, height, i, j;
	double* sequantial_result = NULL, * parallel_result = NULL; //temperature width height
	temperature = atoi(argv[1]);
	width = atoi(argv[2]);
	height = atoi(argv[3]);

	sequantial_result = sequantial(width, height, temperature);
	FILE* fseq = fopen("sequantial_result.txt", "w");
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
			fprintf(fseq, "%7.2lf\t", sequantial_result[i * width + j]);
		fprintf(fseq, "\n");
	}
	printf("Result in sequantial_result.txt\n");

	parallel_result = parallel(width, height, temperature);
	FILE* fpar = fopen("parallel_result.txt", "w");
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
			fprintf(fpar, "%7.2lf\t", parallel_result[i * width + j]);
		fprintf(fpar, "\n");
	}
	printf("Result in parallel_result.txt\n");

		if (sequantial_result != NULL) free(sequantial_result);
		if (parallel_result != NULL) free(parallel_result);
	return 0;
}
