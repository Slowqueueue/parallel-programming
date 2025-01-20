/*26. Найти максимальное простое число, меньшее заданного N,
которое является суммой нескольких последовательно возрастающих простых чисел*/
#include <iostream>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <omp.h>
using namespace std;

void FindPrimeNumbers(int N, vector<bool>& primenums) //находим простые числа
{
    primenums[0] = primenums[1] = false;
    for (int i = 2; i * i <= N; ++i)
        if (primenums[i])
            for (int j = i * i; j <= N; j += i)
                primenums[j] = false;
}

int FindSequential(int N, vector<bool> primenums, vector<int>& summands) //последовательное выполнение
{
    int result = 0;
    for (int i = N; i > 0; i--)
    {
        if (primenums[i])
        {
            result = 0;
            summands.clear();
            for (int j = 2; j < i and result <= i; j++)
            {
                if (result == i) return result;
                if (primenums[j])
                {
                    result += j;
                    summands.push_back(j);
                }
            }
        }
    }
    return 0;
}

int FindParallel(int N, vector<bool>primenums, int& rank, int& size)  //параллельное выполнение
{
    int result = 0;
    for (int i = N - rank; i > 0; i -= size)
    {
        if (primenums[i])
        {
            result = 0;
            for (int j = 2; j < i and result <= i; j++)
            {
                if (result == i) return result;
                if (primenums[j]) result += j;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) //main, args: prog N
{
    if (argc != 2) {
        cout << "Wrong arguments";
        return 0;
    }
    int N = atoi(argv[1]);
    int size, rank;
    int sequentialRes, parallelRes, mpisendBuf;
    double time_start, time_end, time_seq, time_par;
    vector<bool> primenums(N + 1, true);
    vector<int> summands;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    FindPrimeNumbers(N, primenums);

    time_start = omp_get_wtime();
    sequentialRes = FindSequential(N, primenums, summands);
    time_end = omp_get_wtime();
    time_seq = time_end - time_start;

    ofstream fout("summands.txt");
    for (auto i : summands) fout << i << " ";
    fout.close();

    cout << endl << "Sequential:" << endl;
    cout << "Result: " << sequentialRes << endl;
    cout << "Time : " << time_seq << endl << endl;

    time_start = omp_get_wtime();
    mpisendBuf = FindParallel(N, primenums, rank, size);
    MPI_Allreduce(&mpisendBuf, &parallelRes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    time_end = omp_get_wtime();
    time_par = time_end - time_start;

    cout << "Parallel:" << endl;
    cout << "Result: " << parallelRes << endl;
    cout << "Time: " << time_par << endl;

    return 0;
}
