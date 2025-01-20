#include <cmath>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <fstream>
using namespace std;

bool IsTempApproximated(int startingPoint, int endingPoint, int lenX, int lenY, vector<float>& array);
void NoMPICalculate(int lenX, int lenY, vector<float>& array);
void MPICalculate(int processCount, int processRank, int lenX, int lenY, vector<float>& array);

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("Wrong arguments. X Y T\n");
        return -1;
    }
    std::ofstream fout("Result.txt");
    long lenX = atoi(argv[1]), lenY = atoi(argv[2]);
    float Temperature = atoi(argv[3]);
    double start, end, time1, time2;

    if (lenX < 3 || lenY < 3) {
        printf("X and Y must be greater than 3\n");
        return -1;
    }

    MPI_Init(NULL, NULL);
    int processCount, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    vector<float> array1(lenX * lenY, 0), array2(lenX * lenY, 0);

    float step = Temperature / (lenX / 2);
    for (int x = 1; x <= lenX / 2; x++) array2[x * lenY + lenY - 1] = array1[x * lenY + lenY - 1] = step * x;
    step = Temperature / (lenX - lenX / 2 - 1);
    for (int x = lenX / 2 + 1; x < lenX; x++) array2[x * lenY + lenY - 1] = array1[x * lenY + lenY - 1] = Temperature - step * (x - lenX / 2);

    if (processRank == 0)
    {
        printf("Calculating without MPI:\n");

        start = omp_get_wtime();
        NoMPICalculate(lenX, lenY, array1);
        end = omp_get_wtime();

        time1 = end - start;
        printf("Time without MPI: %fs\n", time1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (processRank == 0) printf("Calculating with MPI using %d processes\n", processCount);

    start = omp_get_wtime();
    MPICalculate(processCount, processRank, lenX, lenY, array2);
    end = omp_get_wtime();

    time2 = end - start;
    if (processRank == 0) {
        printf("Time with MPI: %fs\n", time2);
        fout << "[X] [Y] MPI temperature  No MPI temperature\n";

        for (unsigned long long i = 0; i < array1.size(); i++)
        {
            int x = i / lenY, y = i % lenY;
            fout << "[" << x << "] [" << y << "] " << array1[i] << "  " << array2[i] << endl;
        }
        fout.close();
    }
    MPI_Finalize();
    return 0;
}

void NoMPICalculate(int lenX, int lenY, vector<float>& array)
{
    bool IsApproximated = false;
    do {
        IsApproximated = IsTempApproximated(0, array.size(), lenX, lenY, array);
    } while (!IsApproximated);
    array[lenX * lenY / 2] = 0;
    array[lenX * lenY / 2 - 1] = 0;
    array[lenX * lenY / 2 + 1] = 0;
    array[lenX * lenY / 2 - lenX] = 0;
    array[lenX * lenY / 2 + lenX] = 0;
}

bool IsTempApproximated(int startingPoint, int endingPoint, int lenX, int lenY, vector<float>& array)
{
    bool IsApproximated = true;
    for (int i = startingPoint; i < endingPoint; i++)
    {
        int nbrs = 0, y = i % lenY, x = i / lenY;
        float total = 0, Temperature = 0;

        if (y == 0 || y == lenY - 1 || x == 0 || x == lenX - 1) {
            Temperature = array[i];
            if (fabs(array[i] - Temperature) >= 0.000001) IsApproximated = false;
            array[i] = Temperature;
            continue;
        }

        if (y != 0)
        {
            total += array[x * lenY + y - 1];
            nbrs++;
        }
        if (y != lenY - 1)
        {
            total += array[x * lenY + y + 1];
            nbrs++;
        }
        if (x != 0)
        {
            total += array[(x - 1) * lenY + y];
            nbrs++;
        }
        if (x != lenX - 1)
        {
            total += array[(x + 1) * lenY + y];
            nbrs++;
        }
        Temperature = total / nbrs;
        if (fabs(array[i] - Temperature) >= 0.000001) IsApproximated = false;
        array[i] = Temperature;
    }
    return IsApproximated;
}


void MPICalculate(int processCount, int processRank, int lenX, int lenY, vector<float>& array)
{
    int elementsPerProcess = array.size() / processCount + 1;
    int startingPoint = elementsPerProcess * processRank, endingPoint = min(startingPoint + elementsPerProcess, (int)array.size());
    bool IsProcessApproximated = false, IsAllApproximated = false;
    vector<int> Partsizes, Offsets;
    vector<float> temporal_array(lenX * lenY);

    for (int i = 0; i < processCount; i++)
    {
        int start = elementsPerProcess * i, end = min(start + elementsPerProcess, (int)array.size());
        Offsets.push_back(i * elementsPerProcess);
        Partsizes.push_back(end - start);
    }

    do
    {
        IsProcessApproximated = IsTempApproximated(startingPoint, endingPoint, lenX, lenY, array);
        MPI_Allreduce(&IsProcessApproximated, &IsAllApproximated, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        if (!IsProcessApproximated) IsTempApproximated(startingPoint, endingPoint, lenX, lenY, array);
        MPI_Allgatherv(&array[startingPoint], Partsizes[processRank], MPI_INT, &temporal_array.front(), &Partsizes.front(), &Offsets.front(), MPI_INT, MPI_COMM_WORLD);
        array = temporal_array;
    } while (!IsAllApproximated);
    array[lenX * lenY / 2] = 0;
    array[lenX * lenY / 2 - 1] = 0;
    array[lenX * lenY / 2 + 1] = 0;
    array[lenX * lenY / 2 - lenX] = 0;
    array[lenX * lenY / 2 + lenX] = 0;
}