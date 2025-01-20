#include <stdio.h>
#include <omp.h>
#include <ctime>
#include <string.h>
#include <string>
#include <CL/cl.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
using namespace std;

void FindnoOcl(int* result, int N, int M);
void FindOcl(int* result, int N, int M);

int main(int argc, char* argv[])
{
    std::ofstream fout("histogram.txt");
    double start, end, time1, time2;
    if (argc != 3)
    {
        printf("Wrong arguments. N M\n");
        return -1;
    }
    long N = strtol(argv[1], NULL, 10), M = strtol(argv[2], NULL, 10);
    if (N < 2 || M < 2 || N > M)
    {
        printf("N and M must be greater than 1, and M must be greater than N\n");
        return -1;
    }
    int* result1 = (int*)calloc(5000, sizeof(int)), * result2 = (int*)calloc(5000, sizeof(int));

    start = omp_get_wtime();
    FindnoOcl(result1, N, M);
    end = omp_get_wtime();
    time1 = end - start;
    printf("Without OpenCL finished with time: %fs\n", time1);

    start = omp_get_wtime();
    FindOcl(result2, N, M);
    end = omp_get_wtime();
    time2 = end - start;
    printf("OpenCL finished with time: %fs\n", time2);

    fout << "Length: [Quantity without OCL] [Quantity with OCL]\n";
    for (int i = 0; i < 5000; i++) if (result1[i] != 0 && result2[i] != 0) fout << i << " : [" << result1[i] << "] [" << result2[i] << "]\n";
    fout.close();
    printf("Results are written in histogram.txt\n");
    free(result1);
    free(result2);
    return 0;
}

void FindnoOcl(int* result, int N, int M)
{
    for (int i = N; i <= M; i++)
    {
        unsigned int n = i, length = 1;
        while (n != 1)
        {
            length++;
            n = (n % 2 == 0 ? n / 2 : 3 * n + 1);
        }
        result[length]++;
    }
}

struct Ocl
{
    cl_platform_id cpPlatform;
    cl_context context = NULL;
    cl_int error;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_device_id device_id;
    Ocl(const char** sourceCode)
    {
        error = clGetPlatformIDs(1, &cpPlatform, NULL);
        error = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);
        queue = clCreateCommandQueue(context, device_id, 0, &error);
        program = clCreateProgramWithSource(context, 1, sourceCode, NULL, &error);
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (error != CL_SUCCESS) { printf("Create program error."); exit(-1); }
    }
    ~Ocl()
    {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
};

void FindOcl(int* result, int N, int M)
{
    ifstream fin("kerfile.cl");
    string con((istreambuf_iterator<char>(fin)), (istreambuf_iterator<char>()));
    int elementsPerThread = 100;
    size_t lsize = 64, gsize = ceil((float)(M - N + 1) / elementsPerThread / lsize) * lsize;
    char* codefromfile;
    codefromfile = (char*)calloc(con.length() + 1, sizeof(char));
    strcpy(codefromfile, con.data());
    Ocl clProgram((const char**)&codefromfile);
    cl_mem dResult = clCreateBuffer(clProgram.context, CL_MEM_USE_HOST_PTR, 5000 * sizeof(int), result, NULL);
    cl_int err;
    cl_kernel crtker = clCreateKernel(clProgram.program, "kerfunc", &err);
    err = clSetKernelArg(crtker, 0, sizeof(cl_mem), &dResult) | err;
    err = clSetKernelArg(crtker, 1, sizeof(int), &elementsPerThread) | err;
    err = clSetKernelArg(crtker, 2, sizeof(int), &N) | err;
    err = clSetKernelArg(crtker, 3, sizeof(int), &M) | err;
    clEnqueueNDRangeKernel(clProgram.queue, crtker, 1, NULL, &gsize, &lsize, 0, NULL, NULL);
    clFinish(clProgram.queue);
    if (err != CL_SUCCESS) { printf("Kernel startup error."); exit(-1); }
    clEnqueueReadBuffer(clProgram.queue, dResult, CL_TRUE, 0, 5000 * sizeof(int), result, 0, NULL, NULL);
    clReleaseKernel(crtker);
    clReleaseMemObject(dResult);
    free(codefromfile);
}