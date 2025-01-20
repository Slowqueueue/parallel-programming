#include <time.h>
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;

void Quicksort(int* array, int low, int high);
void QuicksortOMP(int* array, int low, int high);
int partition(int array[], int low, int high, int pivot);
void swap(int array[], int pos1, int pos2);

int main(int argc, char* argv[])
{
    ofstream fout("outfile.txt");
    double start, end, time1, time2;
    srand(time(0));
    int size = atoi(argv[1]);
    int* array = (int*)malloc(size * sizeof(int)), * array2 = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) array2[i] = array[i] = rand() % 10000;

    start = omp_get_wtime();
    Quicksort(array, 0, size - 1);
    end = omp_get_wtime();
    time1 = end - start;

    for (int i = 1; i < size; i++) if (array[i] < array[i - 1]) { printf("Sequential array is not sorted\n"); return 0; }
    printf("Sequential array is sorted\n");
    printf("Sequential time: %fs\n", time1);
    fout << "Sequential array\n";

    for (int i = 0; i < size; i++)
    {
        fout << array[i] << " ";
        if (i != 0 && i % 10 == 0) fout << "\n";
    }

#pragma omp parallel shared(array)
    {
#pragma omp single nowait 
        {
            start = omp_get_wtime();
            QuicksortOMP(array2, 0, size - 1);
            end = omp_get_wtime();
        }
    }
    time2 = end - start;

    for (int i = 1; i < size; i++) if (array2[i] < array2[i - 1]) { printf("Parallel array is not sorted\n"); return 0; }
    printf("Parallel array is sorted\n");
    printf("Parrallel time: %fs\n\n", time2);
    fout << "\nParallel array\n";

    for (int i = 0; i < size; i++)
    {
        fout << array2[i] << " ";
        if (i != 0 && i % 10 == 0) fout << "\n";
    }

    if (time1 >= time2) printf("Parallel %fs faster than sequential\n\n", time1 - time2);
    else printf("Parallel %fs slower than sequential\n\n", time2 - time1);
    free(array);
    free(array2);
    fout.close();
    return 0;
}

void Quicksort(int* array, int low, int high) {
    if (low < high) {
        int pivot = array[high], pos = partition(array, low, high, pivot);
        Quicksort(array, low, pos - 1);
        Quicksort(array, pos + 1, high);
    }
}

int partition(int array[], int low, int high, int pivot) {
    int i = low, j = low;
    while (i <= high) {
        if (array[i] > pivot) i++;
        else {
            swap(array, i, j);
            i++;
            j++;
        }
    }
    return j - 1;
}

void swap(int array[], int pos1, int pos2) {
    int temp;
    temp = array[pos1];
    array[pos1] = array[pos2];
    array[pos2] = temp;
}

void QuicksortOMP(int* array, int low, int high) {
    int i = low, j = high, pivot = array[(i + j) / 2];
    while (i < j)
    {
        while (array[i] < pivot) i++;
        while (array[j] > pivot) j--;
        if (i <= j)
        {
            swap(array, i, j);
            i++;
            j--;
        }
    }
#pragma omp task shared(array)
    {
        if (j > low) QuicksortOMP(array, low, j);
    }
#pragma omp task shared(array)
    {
        if (i < high) QuicksortOMP(array, i, high);
    }
#pragma omp taskwait
}