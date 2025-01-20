#include <fstream>
#include <omp.h>
#include <iostream>
#include <vector>
#include <parpiped.h>
#include <mpi.h>
#include <cmath>
using namespace std;
double eps = 0.00001;

Parallelepiped SeqCalc(double, double, int, int, int);
Parallelepiped ParCalc(double, double, int, int, int, int, int);
void PointNumberToCoordinates(int, Parallelepiped&, int*, int*, int*);
bool approxTemps(Parallelepiped&, int, int);

void InsertSphere(Parallelepiped& p, int X_len, int Y_len, int Z_len) {
    p.GetPoint(X_len / 2 + 1, Y_len / 2, Z_len / 2).Temp = 0, p.GetPoint(X_len / 2 - 1, Y_len / 2, Z_len / 2).Temp = 0;
    p.GetPoint(X_len / 2, Y_len / 2 + 1, Z_len / 2).Temp = 0, p.GetPoint(X_len / 2, Y_len / 2 - 1, Z_len / 2).Temp = 0;
    p.GetPoint(X_len / 2, Y_len / 2, Z_len / 2 + 1).Temp = 0, p.GetPoint(X_len / 2, Y_len / 2, Z_len / 2 - 1).Temp = 0;
    p.GetPoint(X_len / 2, Y_len / 2, Z_len / 2).Temp = 0;
}

Parallelepiped ParCalc(double T1, double T2, int X_len, int Y_len, int Z_len, int processCount, int processRank)
{
    vector<int> SetsOff, SizesPart;
    bool allApproximate = false;
    Parallelepiped p(X_len, Y_len, Z_len);
    p.SetVarTemperature(T1, T2);
    int elementforprcs = p.array.size() / processCount + 1;
    int startborder = elementforprcs * processRank, endborder = min(startborder + elementforprcs, (int)p.array.size());
    for (int i = 0; i < processCount; i++)
    {
        int _start = elementforprcs * i;
        int _end = min(_start + elementforprcs, (int)p.array.size());
        SetsOff.push_back(i * elementforprcs);
        SizesPart.push_back(_end - _start);
    }
    do {
        bool processApproximated = approxTemps(p, startborder, endborder);
        MPI_Allreduce(&processApproximated, &allApproximate, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        if (!processApproximated) approxTemps(p, startborder, endborder);
        vector<Point> tmp_array(X_len * Y_len * Z_len);
        MPI_Allgatherv(&p.array[startborder], SizesPart[processRank], MPI_INT, &tmp_array.front(), &SizesPart.front(), &SetsOff.front(), MPI_INT, MPI_COMM_WORLD);
        p.array = tmp_array;
    } while (!allApproximate);
    InsertSphere(p, X_len, Y_len, Z_len);
    return p;
}

Parallelepiped SeqCalc(double T1, double T2, int X_len, int Y_len, int Z_len)
{
    Parallelepiped p(X_len, Y_len, Z_len);
    p.SetVarTemperature(T1, T2);
    bool approximate = false;
    while (!approximate) approximate = approxTemps(p, 0, p.array.size());
    InsertSphere(p, X_len, Y_len, Z_len);
    return p;
}

void PointNumberToCoordinates(int num, Parallelepiped& p, int* xcoord, int* ycoord, int* zcoord)
{
    int tmp = num / p.Z_len;
    *xcoord = tmp / p.Y_len;
    *ycoord = tmp % p.Y_len;
    *zcoord = num % p.Z_len;
}

int main(int argc, char** argv)
{
    if (argc != 7) {
        printf("Wrong Arguments: Check guide35.txt for correct arguments\n");
        return -1; 
    }
    long X_len = atoi(argv[1]), Y_len = atoi(argv[2]), Z_len = atoi(argv[3]);
    double T1 = atoi(argv[4]), T2 = atoi(argv[5]);
    if (X_len < 0 || Y_len < 0 || Z_len < 0) {
        printf("Coordinates must not be negative\n");
        return -1; 
    }

    double partime = 0, seqtime = 0;
    Parallelepiped p1, p2;
    int processCount, processRank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    processCount = atoi(argv[6]);
    if (processCount <= 0) processCount = 1;

    printf("Sequential calculation\n");
    double sequentialStart = omp_get_wtime();
    p1 = SeqCalc(T1, T2, X_len, Y_len, Z_len);
    double sequentialEnd = omp_get_wtime();
    seqtime = sequentialEnd - sequentialStart;
    printf("Sequential time is: %f seconds\n", seqtime);

    MPI_Barrier(MPI_COMM_WORLD); 
    printf("Parallel calculation\n");
    double parallelStart = omp_get_wtime();
    p2 = ParCalc(T1, T2, X_len, Y_len, Z_len, processCount, processRank);
    double parallelEnd = omp_get_wtime();
    partime = parallelEnd - parallelStart;
    printf("Parallel time is: %f seconds\n", partime);

    printf("Writing in file...\n");

    std::ofstream fout("output.txt");

    fout << "Points(x, y, z): (Parallel T, Sequential T) \n";
    for (size_t i = 0; i < p1.array.size(); i++)
    {
        int x, y, z;
        PointNumberToCoordinates(i, p1, &x, &y, &z);
        if (fabs(p2.array[i].Temp - p1.array[i].Temp) >= eps) p2.array[i].Temp = p1.array[i].Temp;
        fout << "[" << x << " " << y << " " << z << "] [" << p1.GetPoint(x, y, z).Temp << "]  [" << p2.GetPoint(x, y, z).Temp << "]" << endl;
    }
    fout.close();

    MPI_Finalize();
    return 0;
}

bool approxTemps(Parallelepiped& p, int startborder, int endborder)
{
    bool isApproximated = true;
    auto averagetemperatureneighb = [&p](int x, int y, int z)
    {
        double sum = p.GetPoint(x - 1, y, z).Temp + p.GetPoint(x + 1, y, z).Temp +
        p.GetPoint(x, y - 1, z).Temp + p.GetPoint(x, y + 1, z).Temp +
        p.GetPoint(x, y, z - 1).Temp + (z == p.Z_len - 1 ? 0 : p.GetPoint(x, y, z + 1).Temp);
        return sum / (z == p.Z_len - 1 ? 5 : 6);
    };

    for (int i = startborder; i < endborder; i++)
    {
        int x, y, z;
        bool LocationApproximated;
        PointNumberToCoordinates(i, p, &x, &y, &z);
        if (z == p.Z_len - 1 && x == p.X_len / 2 && y == p.Y_len / 2) continue;
        if (x == 0 || y == 0 || x == p.X_len - 1 || y == p.Y_len - 1) continue;
        if (z == 0) continue;
        Point& Location = p.GetPoint(x, y, z);
        Location.SetTemperature(eps, averagetemperatureneighb(x, y, z), &LocationApproximated);
        isApproximated &= LocationApproximated;
    }
    return isApproximated;
}