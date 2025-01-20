#include <mpi.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <omp.h>
using namespace std;


struct Location
{
    float T = 0;
    void SetTemp(float newT, bool* isApproximated, float eps)
    {
        float LegacyT = T;
        T = newT;
        *isApproximated = abs(T - LegacyT) < eps;
    }
};


struct PRLD
{
    int X_size = 0, Y_size = 0, Z_size = 0;
    vector<Location> array;
    PRLD() {}
    PRLD(int X_size, int Y_size, int Z_size) : X_size(X_size), Y_size(Y_size), Z_size(Z_size)
    {
        array = vector<Location>(X_size * Y_size * Z_size);
    }
    void SetVarTemps(float T1, float T2)
    {
        GetLocation(X_size / 2, Y_size / 2, Z_size - 1).T = T1;
        float t2_iter = T2 / (Z_size - 1);
        for (int z = 1; z < Z_size; z++)
        {
            for (int x = 0; x < X_size; x++)
            {
                GetLocation(x, 0, z).T = z * t2_iter;
                GetLocation(x, Y_size - 1, z).T = z * t2_iter;
            }
            for (int y = 0; y < Y_size; y++)
            {
                GetLocation(0, y, z).T = z * t2_iter;
                GetLocation(X_size - 1, y, z).T = z * t2_iter;
            }
        }
    }
    Location& GetLocation(int x, int y, int z)
    {
        return array[(x * Y_size + y) * Z_size + z];
    }
    void operator = (const PRLD& p)
    {
        X_size = p.X_size;
        Y_size = p.Y_size;
        Z_size = p.Z_size;
        array = p.array;
    }
};


void LocationNumberToCoordinates(int num, PRLD& p, int* x, int* y, int* z)
{
    *z = num % p.Z_size;
    int tmp = num / p.Z_size;
    *y = tmp % p.Y_size;
    *x = tmp / p.Y_size;
}


bool approximateTemps(PRLD& p, int startId, int endId)
{
    bool isApproximated = true;
    float eps = 0.00001;
    auto getAVGNeighborsT = [&p](int x, int y, int z)
    {
        float sum = p.GetLocation(x - 1, y, z).T + p.GetLocation(x + 1, y, z).T +
            p.GetLocation(x, y - 1, z).T + p.GetLocation(x, y + 1, z).T +
            p.GetLocation(x, y, z - 1).T + (z == p.Z_size - 1 ? 0 : p.GetLocation(x, y, z + 1).T);
        return sum / (z == p.Z_size - 1 ? 5 : 6);
    };
    for (int i = startId; i < endId; i++)
    {
        int x, y, z;
        LocationNumberToCoordinates(i, p, &x, &y, &z);
        if (z == 0) continue;
        if (x == 0 || y == 0 || x == p.X_size - 1 || y == p.Y_size - 1) continue;
        if (z == p.Z_size - 1 && x == p.X_size / 2 && y == p.Y_size / 2) continue;
        Location& Location = p.GetLocation(x, y, z);
        bool LocationApproximated;
        Location.SetTemp(getAVGNeighborsT(x, y, z), &LocationApproximated, eps);
        isApproximated &= LocationApproximated;
    }
    return isApproximated;
}


PRLD Sequential_method(float T1, float T2, int X_size, int Y_size, int Z_size)
{
    PRLD p(X_size, Y_size, Z_size);
    p.SetVarTemps(T1, T2);
    bool approximate = false;
    while (!approximate)
        approximate = approximateTemps(p, 0, p.array.size());
    //Sphere adjustment assuming sphere is located in the center of parallelepiped and have constant radius = 1
    p.GetLocation(X_size / 2, Y_size / 2, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2 - 1, Y_size / 2, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2 + 1, Y_size / 2, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2 - 1, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2 + 1, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2, Z_size / 2 - 1).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2, Z_size / 2 + 1).T = 0;
    return p;
}


PRLD Parallel_method(float T1, float T2, int X_size, int Y_size, int Z_size, int processCount, int processRank)
{
    PRLD p(X_size, Y_size, Z_size);
    p.SetVarTemps(T1, T2);
    int elemsPerProcess = p.array.size() / processCount + 1;
    int start = elemsPerProcess * processRank;
    int end = min(start + elemsPerProcess, (int)p.array.size());
    vector<int> partSizes, offsets;
    for (int i = 0; i < processCount; i++)
    {
        int _start = elemsPerProcess * i;
        int _end = min(_start + elemsPerProcess, (int)p.array.size());
        partSizes.push_back(_end - _start);
        offsets.push_back(i * elemsPerProcess);
    }
    bool allApproximate = false;
    do
    {
        bool processApproximated = approximateTemps(p, start, end);
        MPI_Allreduce(&processApproximated, &allApproximate,
            1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        if (!processApproximated)
            approximateTemps(p, start, end);
        vector<Location> tmp_array(X_size * Y_size * Z_size);
        MPI_Allgatherv(&p.array[start], partSizes[processRank], MPI_INT, &tmp_array.front(),
            &partSizes.front(), &offsets.front(), MPI_INT, MPI_COMM_WORLD);
        p.array = tmp_array;
    } while (!allApproximate);
    //Sphere adjustment assuming sphere is located in the center of parallelepiped and have constant radius = 1
    p.GetLocation(X_size / 2, Y_size / 2, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2 - 1, Y_size / 2, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2 + 1, Y_size / 2, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2 - 1, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2 + 1, Z_size / 2).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2, Z_size / 2 - 1).T = 0;
    p.GetLocation(X_size / 2, Y_size / 2, Z_size / 2 + 1).T = 0;
    return p;
}


void writeResults(PRLD& p1, float time1, PRLD& p2, float time2)
{
    float eps = 0.00001;
    std::ofstream fout("result.txt");
    if (!fout) { printf("Failed to open result.txt\n"); return; }
    float AVGError = 0;
    for (size_t i = 0; i < p1.array.size(); i++)
    {
        if (fabs(p1.array[i].T - p2.array[i].T) >= eps) p1.array[i].T = p2.array[i].T;
        AVGError += fabs(p1.array[i].T - p2.array[i].T);
    }
    AVGError /= p1.array.size();
    fout << "Average error between temperatures: " << AVGError << "\n\n";
    fout << "Parallel time: " << time1 << " sec, Sequential time: " << time2 << " sec\n";
    fout << "Locations: (Parallel temperature, Sequential temperature) \n";
    for (size_t i = 0; i < p1.array.size(); i++)
    {
        int x, y, z;
        LocationNumberToCoordinates(i, p1, &x, &y, &z);

        fout << "[x:" << x << " y:" << y << " z:" << z
            << "] [" << p1.GetLocation(x, y, z).T << "deg]  [" << p2.GetLocation(x, y, z).T << "deg]" << endl;
    }
    fout.close();
}


int main(int argc, char** argv)
{
    if (argc < 6) { printf("Arguments: X_size, Y_size, Z_size, T1, T2, processCount\n"); return -1; }
    long X_size = strtol(argv[1], NULL, 10);
    long Y_size = strtol(argv[2], NULL, 10);
    long Z_size = strtol(argv[3], NULL, 10);
    float T1 = strtof(argv[4], NULL);
    float T2 = strtof(argv[5], NULL);
    if (X_size < 0 || Y_size < 0 || Z_size < 0) { printf("Wrong arguments\n"); return -1; }
    MPI_Init(NULL, NULL); //Инициализирует среду выполнения вызывающего процесса MPI 
    int processCount, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);//Извлекает количество процессов, участвующих в коммуникаторе. В MPI_COMM_WORLD общее кол-во доступных процессов.
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);//Извлекает ранг вызывающего процесса в группе указанного коммуникатора.
    //MPI_COMM_WORLD - Начальный внутриобщий коммуникатор всех процессов.
    if (argc == 7) processCount = atoi(argv[6]);
    if (processCount <= 0) processCount = 1;
    double partime = 0, seqtime = 0;
    PRLD p1, p2;
        printf("Calculation with 1 process...\n");
        double sequeintialStart = omp_get_wtime();
        p1 = Sequential_method(T1, T2, X_size, Y_size, Z_size);
        double sequeintialEnd = omp_get_wtime();
        seqtime = sequeintialEnd - sequeintialStart;
        printf("Sequential time is: %f seconds\n", seqtime);
    MPI_Barrier(MPI_COMM_WORLD); //Инициирует синхронизацию барьеров для всех членов группы.
    if (processRank == 0)
        printf("Calculation with %d processes...\n", processCount);
    double parallelStart = omp_get_wtime();
    p2 = Parallel_method(T1, T2, X_size, Y_size, Z_size, processCount, processRank);
    double parallelEnd = omp_get_wtime();
    partime = parallelEnd - parallelStart;
    printf("Parallel time is: %f seconds\n", partime);
    if (processRank == 0)
    {
        printf("Writing in file...\n");
        writeResults(p2, partime, p1, seqtime);
    }
    MPI_Finalize();//Завершает среду выполнения вызывающего процесса MPI.
    return 0;
}