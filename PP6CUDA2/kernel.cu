#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <omp.h>

using namespace std;

// CUDA kernel to perform the Sieve of Eratosthenes on the GPU
__global__ void resheto(unsigned long long* a, unsigned long long s) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < s + 1) {
        a[idx] = idx;
        for (int p = 2; p < s + 1; p++) {
            if (a[p] != 0) {
                for (int j = p * p; j < s + 1; j += p)
                    a[j] = 0;
            }
        }
    }
}
void resheto1(unsigned long long* a, unsigned long long s) {//функция для создания решета эратосфена
    for (int i = 0; i < s + 1; i++)
        a[i] = i;
    for (int p = 2; p < s + 1; p++)
    {
        if (a[p] != 0)
        {
            //cout << a[p] << endl;
            for (int j = p * p; j < s + 1; j += p)
                a[j] = 0;
        }
    }
}

// CUDA kernel to calculate the sum of array elements on the GPU
__global__ void sumArray(unsigned long long* array, unsigned long long* result, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ unsigned long long sharedSum[];

    if (idx < size) {
        sharedSum[threadIdx.x] = array[idx];
    }
    else {
        sharedSum[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, sharedSum[0]);
    }
}

void show_m(unsigned long long* podpol, unsigned long long k) {//функция показа динамического массива
    for (int i = 0; i < k; i++) {
        cout << podpol[i] << " ";
    }
}

void posled(unsigned long long length, unsigned long long* mass, unsigned long long& x, string S) {//функция для забивания в массив последовательности
    for (int j = 0; j < length; j++) {
        for (int i = 0; i < length - j; i++) {
            mass[0 + x] = stoull(S.substr(0 + i, 1 + j));//*"substr"-функция среза строки*
            x++;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "no arguments!" << endl;
        return 1;
    }

    double begin1 = omp_get_wtime();


    string M;
    bool cc = true;
    unsigned long long N = stoull(argv[1]), m = 0;
    unsigned long long len;
    unsigned long long* mass_ch = new unsigned long long[100];
    double z = 1.1;

    for (int k = 1; cc == true; k++) {
        unsigned long long x = 0, summ = 0;
        double y = 1.1;
        M = to_string(N + k);
        len = M.length();
        posled(len, mass_ch, x, M);

        for (int i = 0; i < x - 1; i++) {
            summ += mass_ch[i];
        }
        unsigned long long* a_device;
        cudaMallocManaged(&a_device, (summ + 1) * sizeof(unsigned long long));
        cudaMemcpy(a_device, mass_ch, (x - 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        // Call the resheto kernel
        int block_size = 256;
        int num_blocks = (summ + block_size - 1) / block_size;
        resheto << <num_blocks, block_size >> > (a_device, summ);
        cudaDeviceSynchronize();

        // Call the sumArray kernel to calculate the sum on the GPU
        unsigned long long* result_device;
        cudaMallocManaged(&result_device, sizeof(unsigned long long));
        sumArray << <1, block_size, block_size * sizeof(unsigned long long) >> > (a_device, result_device, summ + 1);
        cudaDeviceSynchronize();

        unsigned long long result;
        cudaMemcpy(&result, result_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        cudaFree(a_device);
        cudaFree(result_device);

        if (summ == result) {

            cc = true;
        }
        else {
            for (int i = 0; y - int(y) != 0; i++) {
                m = 2 + i;
                y = log(summ) / log(m);
            }
            if (y == 1) {
                cc = true;
                // cout << "Cannot be number in power" << endl;
            }
            else {
                cc = false;
                z = y;
            }
        }
    }
    delete[] mass_ch;
    cout << "Cuda:" << endl;
    cout << "M:" << M << endl;
    cout << "Osnovanie:" << m << endl;
    cout << "Stepen:" << z << endl;

    double end1 = omp_get_wtime();

    double begin2 = omp_get_wtime();
    string M2;
    bool cc2 = true;
    unsigned long long N2 = stoull(argv[1]), m2 = 0;//начальное вводимое значение, m-основание
    unsigned long long len2; //счётчик массива подпоследовательностей (x), len-длина строки, summ-сумма всех посдпослед.
    unsigned long long* mass_ch2 = new unsigned long long[100];//объявление динамического массива подпоследовательностей
    double z2 = 1.1;
    for (int k = 1; cc2 == true; k++) {
        unsigned long long x = 0, summ = 0;
        double y2 = 1.1;
        M2 = to_string(N2 + k);
        len2 = M2.length();
        posled(len2, mass_ch2, x, M2);
        for (int i = 0; i < x - 1; i++) {
            summ += mass_ch2[i];
        }
        unsigned long long* a = new unsigned long long[summ + 1];//динамический массив из простых чисел
        resheto1(a, summ);
        if (summ == a[summ]) {

            cc2 = true;
        }
        else {
            for (int i = 0; y2 - int(y2) != 0; i++) {
                m2 = 2 + i;
                y2 = log(summ) / log(m2);
            }
            if (y2 == 1) {
                cc2 = true;
            }
            else {
                cc2 = false;
                z2 = y2;

            }

        }
        delete[] a;

    }
    delete[] mass_ch2;

    cout << "No cuda:" << endl;
    cout << "M:" << M2 << endl;
    cout << "Osnovanie:" << m2 << endl;
    cout << "Stepen:" << z2 << endl;

    double end2 = omp_get_wtime();

    cout << "CUDA Time in seconds = " << (end1 - begin1) << endl;
    cout << "NO CUDA Time in seconds = " << (end2 - begin2) << endl;

    return 0;
}
