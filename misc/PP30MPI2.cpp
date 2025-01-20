//30. Для любого целого числа n обозначим через p(n) количество способов представления n
//в виде суммы целых положительных чисел (например, p(5) = 6, потому что 5 = 4+1 = 3+2 = 3+1+1 = 2+2+1 = 2+1+1+1 = 1+1+1+1+1).
//Найти минимальное число n, для которого p(n) больше заданного N.
#include <fstream>
#include <climits>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <numeric>
#include <string>
#include <iostream>

using namespace std;
void DecomposeNumber(int n, int* decmpNumber, vector<int>& decmp, vector<vector<int>>* decmposes = 0);

int Calculate(int processCount, int processRank, int minimumDecomps)
{
    int result = INT_MAX;

    for (int i = 5 + processRank, Iter = 1; ; i += processCount, Iter++)
    {
        vector<int> vectordecompos;
        int decompsNumber = 0;

        if (Iter % 5 == 0 && processCount > 1)
        {
            int globalresult, solution;
            MPI_Allreduce(&result, &globalresult, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (globalresult < i) {
                solution = globalresult;
            }
            else {
                solution = INT_MAX;
            }
            MPI_Allreduce(&solution, &globalresult, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (globalresult != INT_MAX)
            {
                if (result == globalresult) {
                    return result;
                }
                else {
                    return INT_MAX;
                }
            }
        }

        if (result != INT_MAX) {
            continue;
        }

        DecomposeNumber(i, &decompsNumber, vectordecompos);

        if (decompsNumber > minimumDecomps)
        {
            result = i;
            if (processCount == 1) {
                return result;
            }
        }
    }
}

void DataPrint(long N, int result1, int result2, double time1, double time2);

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Error. Arguments: N\n");
        return -1;
    }
    long N = atoi(argv[1]);
    int processCount, processRank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    int result1, result2;
    double seqtime = 0, partime = 0, seqstart = 0, seqend = 0, parstart = 0, parend = 0;
    seqstart = omp_get_wtime();
    result1 = Calculate(1, processRank, N);
    seqend = omp_get_wtime();
    seqtime = seqend - seqstart;
    MPI_Barrier(MPI_COMM_WORLD);
    parstart = omp_get_wtime();
    result2 = Calculate(processCount, processRank, N);
    parend = omp_get_wtime();
    partime = parend - parstart;
    if (result2 != INT_MAX && processRank != 0) {
        MPI_Send(&result2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if (processRank == 0 && result2 == INT_MAX) {
        MPI_Recv(&result2, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    DataPrint(N, result1, result2, seqtime, partime);
    MPI_Finalize();
    return 0;
}

void DataPrint(long N, int result1, int result2, double time1, double time2)
{
    int h;
    vector<int> q;
    vector<vector<int>> decomp1, decomp2;

    DecomposeNumber(result1, &h, q, &decomp1);
    decomp1.pop_back();
    DecomposeNumber(result2, &h, q, &decomp2);
    decomp2.pop_back();

    printf("Sequential time: %fs\n", time1);
    printf("Result: p(%d) = %lu\n", result1, decomp1.size());
    printf("Parallel time: %fs\n", time2);
    printf("Result: p(%d) = %lu\n", result2, decomp2.size());

    ofstream outfile("decompositions.txt");
    if (!outfile) {
        printf("Error. Cannot open file\n");
        return;
    }

    outfile << "Decompositions of " << result1 << " and " << result2 << ":\n";
    auto shon1 = decomp1.begin(), shon2 = decomp2.begin();

    while (shon1 != decomp1.end() || shon2 != decomp2.end())
    {
        if (shon1 == decomp1.end()) {
            break;
        }
        for (size_t i = 0; i < (*shon1).size(); i++) {
            outfile << (*shon1)[i] << (i == (*shon1).size() - 1 ? "" : "+");
        }
        shon1++;
        outfile << " | ";
        if (shon2 == decomp2.end()) {
            break;
        }
        for (size_t i = 0; i < (*shon2).size(); i++) {
            outfile << (*shon2)[i] << (i == (*shon2).size() - 1 ? "" : "+");
        }
        shon2++;
        outfile << endl;
    }
    outfile.close();
}

void DecomposeNumber(int n, int* decmpNumber, vector<int>& decmp, vector<vector<int>>* decmposes)
{
    if (n == 0)
    {
        if (decmpNumber != 0) {
            (*decmpNumber)++;
        }
        if (decmposes != 0) {
            decmposes->push_back(decmp);
        }
        return;
    }

    for (int i = 1; i <= n; i++)
    {
        if (decmp.empty() == false && i > decmp.back()) {
            return;
        }
        if (n - i >= 0)
        {
            vector<int> _decomposition = decmp;
            _decomposition.push_back(i);
            DecomposeNumber(n - i, decmpNumber, _decomposition, decmposes);
        }
        else {
            break;
        }
    }
}