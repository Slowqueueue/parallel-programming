#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
using namespace std;

void noOMPsolve(long remsum, long maxvalue, long idx, long& Count) {
    if (remsum == 0) {
        Count++;
        return;
    }
    for (long i = maxvalue; i >= 1; i--) {
        if (i > remsum) continue;
        else noOMPsolve(remsum - i, i, idx + 1, Count);
    }
}

void OMPsolve(long remsum, long maxvalue, long idx, long& Count) {
#pragma omp parallel num_threads(256)
      {
          if (remsum == 0) {
              Count++;
              return;
          }
        for (long i = maxvalue; i >= 1; i--) {
            if (i > remsum) continue;
            else OMPsolve(remsum - i, i, idx + 1, Count);
        }
      }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Wrong arguments. Arguments: N\n";
        return 0;
    }

    long N = atoi(argv[1]);
    long i = 0;
    long Count = -1;
    double start, end, time1, time2;

    start = omp_get_wtime();
    while (Count <= N) {
        Count = -1;
        ++i;
        noOMPsolve(i, i, 1, Count);
    }
    end = omp_get_wtime();

    time1 = end - start;
    printf("P(%d) = %d\n", i, Count);
    printf("n = %d\n", i);
    printf("Not parallel time: %f\n", time1);

    Count = -1;
    i = 0;

    start = omp_get_wtime();
    while (Count <= N) {
        Count = -1;
        ++i;
        OMPsolve(i, i, 1, Count);
    }
    end = omp_get_wtime();
    
    time2 = end - start;
    printf("P(%d) = %d\n", i, Count);
    printf("n = %d\n", i);
    printf("Parallel time: %f\n", time2);

    double diff = abs(time2 - time1);
    cout << (time1 > time2 ? "OMP is faster " : "OMP is slower ") << "with difference " << diff << endl;
    return 0;
}