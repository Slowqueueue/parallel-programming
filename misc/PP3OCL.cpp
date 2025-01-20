#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <CL/cl.h>
#include <omp.h>
#define MAX_SOURCE_SIZE (0x100000)
using namespace std;

void SequentialMethod(int* matrix, int N, int M, int* serial_results);
int ParallelMethod(int* matrix, int N, int M, int* results);

int main(int argc, char* argv[]) {
    ofstream out;
    srand(time(NULL));
    double start, end;
    int N = atoi(argv[1]); //N size of initial matrix
    int M = atoi(argv[2]); //M size of submatrix
    const int Size = N + (64 - N % 64);
    int* matrixinital = (int*)calloc(N * N, sizeof(int));
    int* sequential_results = new int[3];
    int* parallel_results = (int*)calloc(3 * Size, sizeof(int));
    out.open("matrix_inital.txt");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixinital[i * N + j] = rand() % 10;
            out << matrixinital[i * N + j] << " ";
        }
        out << endl;
    }
    out.close();

    start = omp_get_wtime();
    SequentialMethod(matrixinital, N, M, sequential_results);
    end = omp_get_wtime();

    printf("\nMinimal sum: %d\n", sequential_results[0]);
    printf("Index of first element of submatrix: (%d : %d)\n", sequential_results[1], sequential_results[2]);
    printf("Sequential time: %f\n\n", end - start);

    start = omp_get_wtime();
    int index = ParallelMethod(matrixinital, N, M, parallel_results);
    end = omp_get_wtime();

    printf("Minimal sum: %d\n", parallel_results[index]);
    printf("Index of first element of submatrix: (%d : %d)\n", parallel_results[index + 1], parallel_results[index + 2]);
    printf("Parallel time: %f\n\n", end - start);

    printf("Matrix and submatrixes are saved in .txt files");

    free(parallel_results);
    free(matrixinital);
    return 0;
}

void SequentialMethod(int* matrix, int N, int M, int* sequential_results) {
    ofstream out;
    out.open("submatrix_sequential.txt");
    int sum = 0;
    int min_sum = 100000000;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i + M <= N && j + M <= N) {
                sum = 0;
                for (int i1 = 0; i1 < M; i1++)
                    for (int j1 = 0; j1 < M; j1++)
                        if (i1 >= j1) sum += matrix[(i + i1) * N + (j + j1)];
                if (sum < min_sum) {
                    min_sum = sum;
                    sequential_results[0] = sum;
                    sequential_results[1] = i;
                    sequential_results[2] = j;
                }
            }
    int index_i = sequential_results[1];
    int index_j = sequential_results[2];
    for (int i = index_i; i < M + index_i; i++) {
        for (int j = index_j; j < M + index_j; j++) {
            if (i - index_i >= j - index_j) out << matrix[i * N + j] << " ";
        }
        out << endl;
    }
    out.close();

}

int ParallelMethod(int* matrix, int N, int M, int* parallel_results) {
    ofstream out;
    out.open("submatrix_parallel.txt");
    const int Size = N + (64 - N % 64);
    size_t global_item_size = Size;
    size_t local_item_size = 64;
    FILE* fp;
    char* source_str;
    size_t source_size;
    fp = fopen("kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 3 * Size * sizeof(int), NULL, &ret);
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, N * N * sizeof(int), matrix, 0, NULL, NULL);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "KernelFunc", &ret);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(int), &N);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &M);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 3 * Size * sizeof(int), parallel_results, 0, NULL, NULL);

    int min_sum = 10000000;
    int index = 0;
    for (int i = 0; i < 3 * (N - M + 1); i += 3)
        if (parallel_results[i] < min_sum)
        {
            min_sum = parallel_results[i];
            index = i;
        }
    int index_i = parallel_results[index + 1];
    int index_j = parallel_results[index + 2];
    for (int i = index_i; i < M + index_i; i++) {
        for (int j = index_j; j < M + index_j; j++) {
            if (i - index_i >= j - index_j) out << matrix[i * N + j] << " ";
        }
        out << endl;
    }
    out.close();

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return index;
}