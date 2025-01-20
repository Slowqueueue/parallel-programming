/*23. Вычислить методом последовательных приближений последовательность значений температуры заданной точки параллелепипеда
размером k * m * n, имеющего внутри полость в виде сферы. Теплопроводность материала не равна нулю. 
Нижняя грань параллелепипеда имеет постоянную температуру 0. Геометрический центр верхней грани имеет постоянную температуру T1,
температура сторон этой грани постоянна и равна T2.Начальная температура остальных точек параллелепипеда равна T3*/
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

struct Tochka;
struct Parapiped;

using namespace std;

//объявляем структуры
struct Tochka
{
    float Temper = 0;
    void SetTemper(float newTemper, bool* Approx, float epsilon)
    {
        float OldTemper = Temper;
        Temper = newTemper;
        *Approx = abs(Temper - OldTemper) < epsilon;
    }
};

struct Parapiped
{
    int lenghtX = 0;
    int lenghtY = 0;
    int lenghtZ = 0;
    vector<Tochka> arr;
    Parapiped() {}
    Parapiped(int lenghtX, int lenghtY, int lenghtZ) : lenghtX(lenghtX), lenghtY(lenghtY), lenghtZ(lenghtZ)
    {
        arr = vector<Tochka>(lenghtX * lenghtY * lenghtZ);
    }
    void SetVarTempers(float T1, float T2)
    {
        GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ - 1).Temper = T1;
        float stepT2 = T2 / (lenghtZ - 1);
        for (int Z = 1; Z < lenghtZ; Z++)
        {
            for (int X = 0; X < lenghtX; X++)
            {
                GetTochka(X, 0, Z).Temper = Z * stepT2;
                GetTochka(X, lenghtY - 1, Z).Temper = Z * stepT2;
            }
            for (int Y = 0; Y < lenghtY; Y++)
            {
                GetTochka(0, Y, Z).Temper = Z * stepT2;
                GetTochka(lenghtX - 1, Y, Z).Temper = Z * stepT2;
            }
        }
    }
    Tochka& GetTochka(int X, int Y, int Z)
    {
        return arr[(X * lenghtY + Y) * lenghtZ + Z];
    }
    void operator = (const Parapiped& para)
    {
        lenghtX = para.lenghtX;
        lenghtY = para.lenghtY;
        lenghtZ = para.lenghtZ;
        arr = para.arr;
    }
};
//Прототипы функций
Tochka& GetTochka(int, int, int);
Parapiped noMPImethod(float, float, int, int, int);
Parapiped MPImethod(float, float, int, int, int, int, int);
void TochkaNumberToCoord(int, Parapiped&, int*, int*, int*);
void PrintOutfile(Parapiped&, float, Parapiped&, float);
bool approxTempers(Parapiped&, int, int);

int main(int argc, char** argv) //функция main
{
    if (argc < 7) { printf("Args: program, lenghtX, lenghtY, lenghtZ, T1, T2, processCount\n"); return -1; }
    long lenghtX = strtol(argv[1], NULL, 10);
    long lenghtY = strtol(argv[2], NULL, 10);
    long lenghtZ = strtol(argv[3], NULL, 10);
    float T1 = strtof(argv[4], NULL);
    float T2 = strtof(argv[5], NULL);
    if (lenghtX < 0 || lenghtY < 0 || lenghtZ < 0) { printf("Wrong args\n"); return -1; }
    MPI_Init(NULL, NULL);
    int processCount, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    processCount = atoi(argv[6]);
    if (processCount <= 0) processCount = 1;
    double MPItime = 0, noMPItime = 0;
    Parapiped para1, para2;
    printf("Computaion with 1 process...\n");
    double noMPIStart = omp_get_wtime();
    para1 = noMPImethod(T1, T2, lenghtX, lenghtY, lenghtZ);
    double noMPIEnd = omp_get_wtime();
    noMPItime = noMPIEnd - noMPIStart;
    printf("noMPI time is: %f seconds\n", noMPItime);
    MPI_Barrier(MPI_COMM_WORLD);
    if (processRank == 0)
        printf("Computation with %d processes...\n", processCount);
    double MPIStart = omp_get_wtime();
    para2 = MPImethod(T1, T2, lenghtX, lenghtY, lenghtZ, processCount, processRank);
    double MPIEnd = omp_get_wtime();
    MPItime = MPIEnd - MPIStart;
    printf("MPI time is: %f seconds\n", MPItime);
    if (processRank == 0)
    {
        printf("Printing in outfile...\n");
        PrintOutfile(para2, MPItime, para1, noMPItime);
    }
    MPI_Finalize();
    return 0;
}

//Функция реализующая последовательное вычисление
Parapiped noMPImethod(float T1, float T2, int lenghtX, int lenghtY, int lenghtZ) 
{
    Parapiped para(lenghtX, lenghtY, lenghtZ);
    para.SetVarTempers(T1, T2);
    bool approx = false;
    while (!approx)
        approx = approxTempers(para, 0, para.arr.size());
    //Вносим сферу в параллелепипед
    para.GetTochka(lenghtX / 2, lenghtY / 2 - 1, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2 + 1, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ / 2 - 1).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ / 2 + 1).Temper = 0;
    para.GetTochka(lenghtX / 2 - 1, lenghtY / 2, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2 + 1, lenghtY / 2, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ / 2).Temper = 0;
    return para;
}

//Функция реализующая параллельное вычисление
Parapiped MPImethod(float T1, float T2, int lenghtX, int lenghtY, int lenghtZ, int processCount, int processRank)
{
    Parapiped para(lenghtX, lenghtY, lenghtZ);
    para.SetVarTempers(T1, T2);
    int elementsPerProcess = para.arr.size() / processCount + 1;
    int start = elementsPerProcess * processRank;
    int end = min(start + elementsPerProcess, (int)para.arr.size());
    vector<int> PartSizes, OffSets;
    for (int i = 0; i < processCount; i++)
    {
        int _start = elementsPerProcess * i;
        int _end = min(_start + elementsPerProcess, (int)para.arr.size());
        PartSizes.push_back(_end - _start);
        OffSets.push_back(i * elementsPerProcess);
    }
    bool allApprox = false;
    do
    {
        bool processApprox = approxTempers(para, start, end);
        MPI_Allreduce(&processApprox, &allApprox,
            1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        if (!processApprox)
            approxTempers(para, start, end);
        vector<Tochka> temp_array(lenghtX * lenghtY * lenghtZ);
        MPI_Allgatherv(&para.arr[start], PartSizes[processRank], MPI_INT, &temp_array.front(),
            &PartSizes.front(), &OffSets.front(), MPI_INT, MPI_COMM_WORLD);
        para.arr = temp_array;
    } while (!allApprox);
    //Вносим сферу в параллелепипед
    para.GetTochka(lenghtX / 2, lenghtY / 2 - 1, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2 + 1, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ / 2 - 1).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ / 2 + 1).Temper = 0;
    para.GetTochka(lenghtX / 2 - 1, lenghtY / 2, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2 + 1, lenghtY / 2, lenghtZ / 2).Temper = 0;
    para.GetTochka(lenghtX / 2, lenghtY / 2, lenghtZ / 2).Temper = 0;
    return para;
}

//Функция для записи в файл
void PrintOutfile(Parapiped& para1, float time1, Parapiped& para2, float time2)
{
    std::ofstream fout("Outfile.txt");
    if (!fout) { printf("Error opening Outfile.txt\n"); return; }
    float averagerror = 0;
    float epsilon = 0.00001;
    for (size_t i = 0; i < para1.arr.size(); i++)
    {
        if (fabs(para1.arr[i].Temper - para2.arr[i].Temper) >= epsilon) para1.arr[i].Temper = para2.arr[i].Temper;
        averagerror += fabs(para1.arr[i].Temper - para2.arr[i].Temper);
    }
    averagerror /= para1.arr.size();
    fout << "Average error is: " << averagerror << "\n\n";
    fout << "MPI time: " << time1 << " sec, noMPI time: " << time2 << " sec\n";
    fout << "Tochki: (MPI temperature, noMPI temperature) \n";
    for (size_t i = 0; i < para1.arr.size(); i++)
    {
        int X, Y, Z;
        TochkaNumberToCoord(i, para1, &X, &Y, &Z);
        fout << "[x:" << X << " y:" << Y << " z:" << Z
            << "] [" << para1.GetTochka(X, Y, Z).Temper << "]  [" << para2.GetTochka(X, Y, Z).Temper << "]" << endl;
    }
    fout.close();
}

//Функция для аппроксимации температур
bool approxTempers(Parapiped& para, int start, int end)
{
    bool Approx = true;
    float epsilon = 0.00001;
    auto averageneighborsTemper = [&para](int X, int Y, int Z)
    {
        float Sum = para.GetTochka(X - 1, Y, Z).Temper + para.GetTochka(X + 1, Y, Z).Temper +
            para.GetTochka(X, Y - 1, Z).Temper + para.GetTochka(X, Y + 1, Z).Temper +
            para.GetTochka(X, Y, Z - 1).Temper + (Z == para.lenghtZ - 1 ? 0 : para.GetTochka(X, Y, Z + 1).Temper);
        return Sum / (Z == para.lenghtZ - 1 ? 5 : 6);
    };
    for (int i = start; i < end; i++)
    {
        int X, Y, Z;
        TochkaNumberToCoord(i, para, &X, &Y, &Z);
        if (Z == 0) continue;
        if (X == 0 || Y == 0 || X == para.lenghtX - 1 || Y == para.lenghtY - 1) continue;
        if (Z == para.lenghtZ - 1 && X == para.lenghtX / 2 && Y == para.lenghtY / 2) continue;
        Tochka& Tochka = para.GetTochka(X, Y, Z);
        bool TochkaApprox;
        Tochka.SetTemper(averageneighborsTemper(X, Y, Z), &TochkaApprox, epsilon);
        Approx &= TochkaApprox;
    }
    return Approx;
}

//Функция для перевода номера точки в координаты
void TochkaNumberToCoord(int num, Parapiped& para, int* X, int* Y, int* Z)
{
    *Z = num % para.lenghtZ;
    int temp = num / para.lenghtZ;
    *Y = temp % para.lenghtY;
    *X = temp / para.lenghtY;
}