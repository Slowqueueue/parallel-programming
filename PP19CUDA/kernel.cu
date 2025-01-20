#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef unsigned long long ull;

__global__ void CalculateParallel(unsigned* Results, unsigned long long N, unsigned long long Add) //Функция для параллельного расчета.
{
    int n = threadIdx.x + blockIdx.x * blockDim.x + Add;
    if (n >= N) { return; }
    unsigned Steps = 0;
    for (unsigned long long n2 = n; n2 != 1; ++Steps)
        n2 = (n2 % 2 == 0 ? n2 / 2 : 3 * n2 + 1);
    Results[n - Add] = Steps;
}

void CalculateWithCuda(vector<unsigned>& Results, unsigned long long N) //Функция для запуска параллельных вычислений
{
    if (cudaSetDevice(0) != cudaSuccess) throw "CUDA initialization error";
    cudaDeviceProp propert;
    if (cudaGetDeviceProperties(&propert, 0) != cudaSuccess) throw "CUDA get device properties error";
    const int max_threads = propert.maxThreadsDim[0];
    const int max_blocks = min(int((1024 / sizeof(unsigned)) * 1024 * 1024 / max_threads), propert.maxGridSize[0]);
    const int max_total = max_threads * max_blocks;
    unsigned long long max_n = N - 1;
    Results.resize(max_n);
    unsigned* d_Results;
    if (cudaMalloc((void**)&d_Results, min(max_n, ull(max_total)) * sizeof(unsigned)) != cudaSuccess) throw "CUDA memory allocation error";
    for (unsigned long long Add = 1; Add < N; Add += max_total) {
        unsigned long long threads = min(N - Add, ull(max_total));
        int Nb_threads = (threads < max_threads) ? threads : max_threads;
        int Nb_blocks = (threads - 1) / max_threads + 1;
        CalculateParallel << <Nb_blocks, Nb_threads >> > (d_Results, N, Add);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(d_Results);
            throw "CUDA kernel error";
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(d_Results);
            throw "CUDA synchronization error";
        }
        if (cudaMemcpy(Results.data() + Add - 1, d_Results, threads * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(d_Results);
            throw "CUDA memory copy error";
        }
    }
    cudaFree(d_Results);
}