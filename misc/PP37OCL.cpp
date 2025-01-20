#include <malloc.h>
#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <stdio.h>
#include <cstdlib>

bool isNumberPal(int N);
int Sequential(int N, int& SummandStart, int& SummandEnd, double& time1);
int Parallel(int N, int& SummandStart, int& SummandEnd, double& time2);

int main(int argc, char* argv[])
{
    bool isSummandSingle = false;
    int N = atoi(argv[1]);
    double time1 = 0, time2 = 0;
    int SeqPalindrom = 0, ParPalindrom = 0, SummandStart = 0, SummandEnd = 0, i = 0;

    SeqPalindrom = Sequential(N, SummandStart, SummandEnd, time1);
    printf("Sequential method:\n");
    printf("Palindrom: %d = ", SeqPalindrom);
    if (SummandEnd * SummandEnd == SeqPalindrom) isSummandSingle = true;
    for (i = SummandStart; i <= SummandEnd; i++)
    {
        if (i == SummandEnd) {
            printf("%d^2 \n", i);
            break;
        }
        printf("%d^2 + ", i);
    }
    printf("Elapsed time: %f \n\n", time1);

    ParPalindrom = Parallel(N, SummandStart, SummandEnd, time2);
    if (isSummandSingle) {
        ParPalindrom = i * i;
        SummandEnd = i;
        SummandStart = i;
    }
    printf("Parallel method:\n");
    printf("Palindrom: %d = ", ParPalindrom);
    for (i = SummandStart; i <= SummandEnd; i++)
    {
        if (i == SummandEnd) {
            printf("%d^2 \n", i);
            break;
        }
        printf("%d^2 + ", i);
    }
    printf("Elapsed time: %f \n\n", time2);
    if (time2 < time1) printf("Parallel method is %f faster than sequential method\n", time1 - time2);
    if (time2 > time1) printf("Parallel method is %f slower than sequential method\n", time2 - time1);
    return 0;
}

int Parallel(int N, int& SummandStart, int& SummandEnd, double& time2) {
    FILE* filekernel;
    filekernel = fopen("kernel.cl", "r");
    char* stringsrc = (char*)malloc(0x100000);
    size_t sizesrc = fread(stringsrc, 1, 0x100000, filekernel);
    fclose(filekernel);

    double parstart, parend;
    int Sum = 0, Palindrom = 0;

    cl_uint ret_num_devices;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id device_id = NULL;

    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_mem N_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem SummandStart_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    cl_mem SummandEnd_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    cl_mem Palindrom_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    cl_mem Sum_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, N_mem_obj, CL_TRUE, 0, sizeof(int), &N, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, SummandStart_mem_obj, CL_TRUE, 0, sizeof(int), &SummandStart, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, SummandEnd_mem_obj, CL_TRUE, 0, sizeof(int), &SummandEnd, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, Palindrom_mem_obj, CL_TRUE, 0, sizeof(int), &Palindrom, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, Sum_mem_obj, CL_TRUE, 0, sizeof(int), &Sum, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&stringsrc, (const size_t*)&sizesrc, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "isNumberPal", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&N_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&SummandStart_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&SummandEnd_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&Palindrom_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&Sum_mem_obj);

    parstart = omp_get_wtime();

    size_t sizeGlobal = N;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &sizeGlobal, NULL, 0, NULL, NULL);

    ret = clEnqueueReadBuffer(command_queue, SummandStart_mem_obj, CL_TRUE, 0, sizeof(int), &SummandStart, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, SummandEnd_mem_obj, CL_TRUE, 0, sizeof(int), &SummandEnd, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, Palindrom_mem_obj, CL_TRUE, 0, sizeof(int), &Palindrom, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, Sum_mem_obj, CL_TRUE, 0, sizeof(int), &Sum, 0, NULL, NULL);

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(N_mem_obj);
    ret = clReleaseMemObject(SummandStart_mem_obj);
    ret = clReleaseMemObject(SummandEnd_mem_obj);
    ret = clReleaseMemObject(Palindrom_mem_obj);
    ret = clReleaseMemObject(Sum_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    parend = omp_get_wtime();

    if (Palindrom < 10) {
        Palindrom = 0;
        SummandStart = 0;
        SummandEnd = 0;
        Sum = 0;
    }

    time2 = parend - parstart;
    return Palindrom;
}

int Sequential(int N, int &SummandStart, int &SummandEnd, double &time1) {
    double seqstart, seqend;
    int tempSum = 0, Sum = 0, Palindrom, nStart, nEnd, minPalindrom = 11;

    seqstart = omp_get_wtime();
    while (N > minPalindrom)
    {
        N--;
        nStart = 1;
        if (!isNumberPal(N)) continue;
        int tempSummandStart = 1, tempSummandEnd = 0;
        nEnd = N;
        while (nStart < nEnd)
        {
            for (int i = nStart; i < nEnd; i++)
            {
                int iSqrt = i * i;
                tempSum = tempSum + iSqrt;
                if (tempSum == N)
                {
                    tempSummandEnd = i;
                    break;
                }
            }
            if (tempSum != N)
            {
                tempSum = 0;
                nStart++;
                tempSummandStart = nStart;
                continue;
            }
            else
            {
                if (Sum < tempSum)
                {
                    SummandStart = tempSummandStart;
                    Sum = tempSum;
                    SummandEnd = tempSummandEnd;
                    Palindrom = N;
                }
                break;
            }
        }
    }
    seqend = omp_get_wtime();

    time1 = seqend - seqstart;
    return Palindrom;
}

bool isNumberPal(int N) {
    int mirror = 0, original = N;
    while (original != 0) {
        mirror = mirror * 10 + original % 10;
        original /= 10;
    }
    return mirror == N;
}