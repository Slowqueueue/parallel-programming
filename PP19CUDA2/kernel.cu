#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
using namespace std;

unsigned long long Parallel(unsigned long long N, vector<unsigned>& Results);
unsigned long long Sequential(unsigned long long N, vector<unsigned>& Results);
__global__ void ParallelKernel(unsigned* Results, unsigned long long N, unsigned long long Add);

int main(int argc, char* argv[]) {
    vector<unsigned> Results;
    unsigned long long n, N = atoi(argv[1]);
    double seqstart, seqend, parstart, parend, time1, time2;
    if (N < 2) {
        printf("N must be 2 or more");
        return 0;
    }
    Results.reserve(N - 1);

    seqstart = omp_get_wtime();
    n = Sequential(N, Results);
    seqend = omp_get_wtime();

    time1 = seqend - seqstart;
    printf("Sequential method:\n");
    printf("Result: %llu with %llu steps\n", n, Results[n - 1]);
    printf("Time: %f\n\n", time1);

    parstart = omp_get_wtime();
    n = Parallel(N, Results);
    parend = omp_get_wtime();

    time2 = parend - parstart;
    printf("Parallel method:\n");
    printf("Result: %llu with %llu steps\n", n, Results[n - 1]);
    printf("Time: %f\n\n", time2);

    if (time2 < time1) printf("Parallel method is %f faster than sequential method\n", time1 - time2);
    if (time2 > time1) printf("Parallel method is %f slower than sequential method\n", time2 - time1);
    return 0;
}

unsigned long long Parallel(unsigned long long N, vector<unsigned>& Results) {
    cudaError_t cudaStatus;
    cudaDeviceProp propert;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        return 0;
    }
    if (cudaGetDeviceProperties(&propert, 0) != cudaSuccess) {
        fprintf(stderr, "CUDA get device properties error");
        return 0;
    }

    const int max_threads = propert.maxThreadsDim[0];
    const int max_blocks = min(int((1024 / sizeof(unsigned)) * 1024 * 1024 / max_threads), propert.maxGridSize[0]);
    unsigned long long max_total = max_threads * max_blocks;
    unsigned long long max_n = N - 1;
    Results.resize(max_n);
    unsigned* d_Results;

    if (cudaMalloc((void**)&d_Results, min(max_n, max_total) * sizeof(unsigned)) != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation error");
        return 0;
    }

    for (unsigned long long Add = 1; Add < N; Add += max_total) {
        unsigned long long threads = min(N - Add, max_total);
        int Nb_threads = (threads < max_threads) ? threads : max_threads;
        int Nb_blocks = (threads - 1) / max_threads + 1;

        ParallelKernel KERNEL_ARGS2(Nb_blocks, Nb_threads) (d_Results, N, Add);

        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(d_Results);
            fprintf(stderr, "CUDA kernel error");
            return 0;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(d_Results);
            fprintf(stderr, "CUDA synchronization error");
            return 0;
        }
        if (cudaMemcpy(Results.data() + Add - 1, d_Results, threads * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(d_Results);
            fprintf(stderr, "CUDA memory copy error");
            return 0;
        }
    }
    cudaFree(d_Results);
    return distance(Results.begin(), max_element(Results.begin(), Results.end())) + 1;
}

unsigned long long Sequential(unsigned long long N, vector<unsigned>& Results) {
    unsigned long long m, n, Elements;
    for (m = 1; m < N; m++) {
        Elements = 0;
        for (n = m; n != 1; Elements++) n = n % 2 == 0 ? n / 2 : 3 * n + 1;
        Results.push_back(Elements);
    }
    return distance(Results.begin(), max_element(Results.begin(), Results.end())) + 1;
}

__global__ void ParallelKernel(unsigned* Results, unsigned long long N, unsigned long long Add) {
    int m = threadIdx.x + blockIdx.x * blockDim.x + Add;
    unsigned long long n, Elements;
    if (m >= N) return;
    Elements = 0;
    for (n = m; n != 1; Elements++) n = (n % 2 == 0 ? n / 2 : 3 * n + 1);
    Results[m - Add] = Elements;
}