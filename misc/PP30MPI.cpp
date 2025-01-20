#include <numeric>
#include <string>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <climits>
using namespace std;

#define NO_RESULT INT_MAX
#define ITERATION_SYNCHRONIZE 5

int Find(int minDecomposes, int processCount, int processRank);
void printRes(int result1, double time1, int result2, double time2, long N);

int main(int argc, char** argv)
{
    double time1 = 0, time2 = 0;
    int result1, result2;
    int processCount, processRank;
    if (argc != 2) {
        printf("Wrong arguments. Args: N\n");
        return -1;
    }
    long N = strtol(argv[1], NULL, 10);
    if (N < 2) {
        printf("Wrong arg\n");
        return -1;
    }

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    auto calculate = [N, processRank](int(*func)(int, int, int), int processCount, int* res)
    {
        double start, end;
        start = omp_get_wtime();
        *res = func(N, processCount, processRank);
        end = omp_get_wtime();
        return (end - start);
    };

    time1 = calculate(Find, 1, &result1);

    MPI_Barrier(MPI_COMM_WORLD);
    time2 = calculate(Find, processCount, &result2);

    if (result2 != NO_RESULT && processRank != 0) MPI_Send(&result2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (processRank == 0 && result2 == NO_RESULT) MPI_Recv(&result2, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printRes(result2, time2, result1, time1, N);

    MPI_Finalize();
    return 0;
}

void Decompose(int n, vector<int>& decompos, int* decompNumber = nullptr, vector<vector<int>>* decomposes = nullptr)
{
    if (n == 0)
    {
        if (decomposes != nullptr) decomposes->push_back(decompos);
        if (decompNumber != nullptr) (*decompNumber)++;
        return;
    }
    for (int i = 1; i <= n; i++)
    {
        if (decompos.empty() == false && i > decompos.back()) return;
        if (n - i >= 0)
        {
            vector<int> _decomposition = decompos;
            _decomposition.push_back(i);
            Decompose(n - i, _decomposition, decompNumber, decomposes);
        }
        else break;
    }
}

void printRes(int result1, double time1, int result2, double time2, long N)
{
    std::ofstream fout("decs.txt");
    if (!fout) { printf("fail to open result.txt\n"); return; }
    vector<vector<int>> dec1, dec2;
    auto decompos = [](int digit, vector<vector<int>>& decs)
    {
        int s;
        vector<int> d;
        Decompose(digit, d, &s, &decs);
        decs.pop_back();
    };
    decompos(result1, dec1);
    decompos(result2, dec2);
    auto writeResult = [&fout](int res, double time, vector<vector<int>>& decs)
    {
        cout << "Time: " << time << "c\n";
        cout << "Result: p(" << res << ") = " << decs.size() << "\n";
    };
    cout << "MPI:\n";
    writeResult(result1, time1, dec1);
    cout << "\nWithout MPI:\n";
    writeResult(result2, time2, dec2);
    cout << endl << result1 << (result1 == result2 ? "=" : "!=") << result2 << ", " << (result1 == result2 ? "Identical!" : "Not identical!") << "\n";
    if (N > 100000) return;
    fout << "Decompositions of " << result1 << " and " << result2 << ":\n";
    auto iter1 = dec1.begin(), iter2 = dec2.begin();
    auto writeDecompos = [&fout](vector<vector<int>>::iterator& iter, vector<vector<int>>::iterator end)
    {
        if (iter == end) return;
        for (size_t i = 0; i < (*iter).size(); i++)
            fout << (*iter)[i] << (i == (*iter).size() - 1 ? "" : "+");
        iter++;
    };
    while (iter1 != dec1.end() || iter2 != dec2.end())
    {
        writeDecompos(iter1, dec1.end());
        fout << " : ";
        writeDecompos(iter2, dec2.end());
        fout << endl;
    }
    fout.close();
}

int Find(int minDecomposes, int processCount, int processRank)
{
    int resNumber = NO_RESULT;
    auto synchronizeresGlobal = [&resNumber](int curNumberToHandle)
    {
        int resGlobal;
        MPI_Allreduce(&resNumber, &resGlobal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        int answer = (resGlobal < curNumberToHandle ? resGlobal : NO_RESULT);
        MPI_Allreduce(&answer, &resGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return resGlobal;
    };
    for (int i = 5 + processRank, iteration = 1; ; i += processCount, iteration++)
    {
        if (iteration % ITERATION_SYNCHRONIZE == 0 && processCount > 1)
        {
            int resGlobal = synchronizeresGlobal(i);
            if (resGlobal != NO_RESULT)
            {
                if (resNumber == resGlobal) return resNumber;
                else return NO_RESULT;
            }
        }
        if (resNumber != NO_RESULT) continue;
        vector<int> decompos;
        int decompNumber = 0;
        Decompose(i, decompos, &decompNumber);
        if (decompNumber > minDecomposes)
        {
            resNumber = i;
            if (processCount == 1) return resNumber;
        }
    }
}