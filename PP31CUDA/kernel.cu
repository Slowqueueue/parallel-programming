#define _CRT_SECURE_NO_WARNINGS
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define BLOCK_SIZE 32
#define ll long long

cudaError_t CudaCalcualtions(ll N, ll* a, ll* b, ll size);
__global__ void CalculationsKernel(ll N, ll* a, ll* b);

int main(int argc, char** argv)
{
    FILE* fsimp;
    char fname[32] = "simple.txt";
    int N;
    N = atoi(argv[1]);

    if ((fsimp = fopen(fname, "r")) == NULL)
    {
        printf("File not found");
        getchar();
        return 0;
    }
    ll N1;
    fscanf(fsimp, "%lld", &N1);
    ll* a = (ll*)malloc(N1 * sizeof(ll));
    for (int i = 0; i < N1; i++) {
        fscanf(fsimp, "%lld", (a + i));
    }
    fclose(fsimp);

    double startpar, startseq, endpar, endseq;
    ll* b = (ll*)malloc(N * 3 * sizeof(ll));

    startpar = omp_get_wtime();
    cudaError_t cudaStatus = CudaCalcualtions(N, a, b, N1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    ll num_1_find = 0, num_2_find = 0, num_3_find = 0, n_find = 0;
    for (ll n = 0; n < N; n++) {
        if (*(b + n * 3) != 0) {
            num_1_find = *(b + n * 3);
            num_2_find = *(b + n * 3 + 1);
            num_3_find = *(b + n * 3 + 2);
            n_find = n;
        }
    }
    endpar = omp_get_wtime();

    startseq = omp_get_wtime();
    int flag1 = 0, flag2 = 0, flag3 = 0;
    ll i = 0, j = 0, k = 0;
    ll num_1, num_2, num_3;
    ll num_1_find_l = 0, num_2_find_l = 0, num_3_find_l = 0, n_find_l = 0;
    for (ll n = 0; n < N; n++) {
        i = 0;
        while (flag1 == 0) {
            num_1 = *(a + i);
            if (num_1 * num_1 + 8 + 16 > n) break;
            j = 0;
            while (flag2 == 0) {
                num_2 = *(a + j);
                if (num_1 * num_1 + num_2 * num_2 * num_2 + 16 > n) break;
                k = 0;
                while (flag3 == 0) {
                    num_3 = *(a + k);
                    if ((num_1 * num_1 + num_2 * num_2 * num_2 + num_3 * num_3 * num_3 * num_3) > n) break;
                    if ((num_1 * num_1 + num_2 * num_2 * num_2 + num_3 * num_3 * num_3 * num_3) == n) {
                        num_1_find_l = num_1, num_2_find_l = num_2, num_3_find_l = num_3;
                        flag1 = 1, flag2 = 1, flag3 = 1;
                        n_find_l = n;
                    }
                    k++;
                }
                j++;
            }
            i++;
        }
        flag1 = 0, flag2 = 0, flag3 = 0;
    }
    endseq = omp_get_wtime();

    if (num_1_find_l != num_1_find || num_2_find_l != num_2_find || num_3_find_l != num_3_find || n_find_l != n_find) {
        printf("Error. Wrong calculations\n");
        return 0;
    }
    if (n_find != 0) printf("Result: %lld^2 + %lld^3 + %lld^4 = %lld\n", num_1_find, num_2_find, num_3_find, n_find);
    else printf("\nError. No such numbers");

    printf("\nParallel time: %f s", endpar - startpar);
    printf("\nSequential time: %f s\n", endseq - startseq);
    free(a);
    free(b);
    return 0;
}

cudaError_t CudaCalcualtions(ll N, ll* a, ll* b, ll size)
{
    ll* dev_a = 0;
    ll* dev_b = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(ll));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * 3 * sizeof(ll));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(ll), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    int blocks = (N + BLOCK_SIZE - N % BLOCK_SIZE) / BLOCK_SIZE;
    CalculationsKernel KERNEL_ARGS2(blocks, BLOCK_SIZE) (N, dev_a, dev_b);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(b, dev_b, N * 3 * sizeof(ll), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cudaStatus;
}

__global__ void CalculationsKernel(ll N, ll* a, ll* b)
{
    ll n = blockDim.x * blockIdx.x + threadIdx.x;
    int fl1 = 0, fl2 = 0, fl3 = 0;
    ll i = 0, j = 0, k = 0;
    ll num1, num2, num3;
    ll num1_find = 0, num2_find = 0, num3_find = 0;
    if (n < N) {
        while (fl1 == 0) {
            num1 = *(a + i);
            if (num1 * num1 + 8 + 16 > n) break;
            j = 0;
            while (fl2 == 0) {
                num2 = *(a + j);
                if (num1 * num1 + num2 * num2 * num2 + 16 > n) break;
                k = 0;
                while (fl3 == 0) {
                    num3 = *(a + k);
                    if ((num1 * num1 + num2 * num2 * num2 + num3 * num3 * num3 * num3) > n) break;
                    if ((num1 * num1 + num2 * num2 * num2 + num3 * num3 * num3 * num3) == n) {
                        fl1 = 1, fl2 = 1, fl3 = 1;
                        num1_find = num1, num2_find = num2, num3_find = num3;
                    }
                    k++;
                }
                j++;
            }
            i++;
        }
    }
    *(b + 3 * n) = num1_find;
    *(b + 3 * n + 1) = num2_find;
    *(b + 3 * n + 2) = num3_find;
}