#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#define TAG_SUCCESS 97
#define TAG_CUR 96
#define TAG_PREV 95
#define TAG_INITIATIVE 94
#define TAG_STAND 93
using namespace std;

void ReadMessages(int rank, void* s_current, void* Success, void* s_previous, void* ready, void* standby);
bool IsSimple(long num);

int main(int argc, char* argv[]) {
	if (argc != 2) {
		printf("Wrong arguments. Arguments: N");
		return 0;
	}

	long n = atoi(argv[1]);
	bool found = false;
	long avg = n + n % 2;
	double startseq, endseq, startpar, endpar;

	printf("Sequential method: \n");
	startseq = omp_get_wtime();
	while (!found)
	{
		if (IsSimple(avg + 1) && IsSimple(avg - 1))
			for (long i = avg - 2; i > 2; i -= 2)
				if (IsSimple(i + 1) && IsSimple(i - 1))
				{
					if ((avg - i > n))
					{
						printf("Twins: %ld-%ld and %ld-%ld\n", i - 1, i + 1, avg - 1, avg + 1);
						found = true;
					}
					break;
				}
		avg += 2;
	}
	endseq = omp_get_wtime();
	printf("Sequential time: %f s\n\n", endseq - startseq);

	printf("Parallel method\n");
	int errCode;
	unsigned long i = 0, N = 0;
	if ((errCode = MPI_Init(&argc, &argv)) != 0) {
		printf("Error occurred: %d\n", errCode);
		return 0;
	}

	int rank, size;
	errCode = MPI_Comm_size(MPI_COMM_WORLD, &size);
	errCode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	N = atoi(argv[1]);

	startpar = omp_get_wtime();
	MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	int Success = 0, one = 1;
	unsigned long l_nil = 0, s_current = 0, s_previous = 4;
	for (i = rank + 1; Success == 0; i += size) {
		unsigned long a = (6 * i - 1), b = (6 * i + 1);
		int ready = 0, standby = 0;
		if (IsSimple(a) && IsSimple(b)) {
			unsigned long centre = a + 1;
			if (s_current == 0) {
				s_current = centre;
				for (int i = 0; i < size; i++)
					if (i != rank) MPI_Send(&centre, 1, MPI_UNSIGNED_LONG, i, TAG_CUR, MPI_COMM_WORLD);
				if ((s_current - s_previous) > N) {
					Success = 1;
					for (int i = 0; i < size; i++)
						if (i != rank) MPI_Send(&one, 1, MPI_INT, i, TAG_SUCCESS, MPI_COMM_WORLD);
				}
				else {
					s_previous = s_current;
					s_current = l_nil;
					for (int i = 0; i < size; i++) {
						if (i != rank) {
							MPI_Send(&s_previous, 1, MPI_UNSIGNED_LONG, i, TAG_PREV, MPI_COMM_WORLD);
							MPI_Send(&s_current, 1, MPI_UNSIGNED_LONG, i, TAG_CUR, MPI_COMM_WORLD);
						}
					}
				}
			}
		}
		while (ready == 0) {
			if (rank == 0) ready = 1;
			else ReadMessages(rank, &s_current, &Success, &s_previous, &ready, &standby);
		}

		if (rank + 1 < size) MPI_Send(&one, 1, MPI_INT, (rank + 1), TAG_INITIATIVE, MPI_COMM_WORLD);
		else {
			standby = 1;
			for (int i = 0; i < size; i++)
				if (i != rank) MPI_Send(&standby, 1, MPI_INT, i, TAG_STAND, MPI_COMM_WORLD);
		}

		while (standby == 0) ReadMessages(rank, &s_current, &Success, &s_previous, &ready, &standby);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	endpar = omp_get_wtime();

	unsigned long a = s_previous - 1, b = s_previous + 1;
	printf("Twins: %ld-%ld and %ld-%ld\n", a, b, (s_current - 1), (s_current + 1));
	printf("Parallel time: %f s\n", endpar - startpar);

	MPI_Finalize();
	return 0;
}

bool IsSimple(long num)
{
	int limit = (int)sqrt(num);
	for (int j = 3; j <= limit; j += 2)
		if (num % j == 0) return false;
	return true;
};

void ReadMessages(int rank, void* s_current, void* Success, void* s_previous, void* ready, void* standby) {
	MPI_Status status;
	MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	if (status.MPI_TAG == TAG_CUR) {
		MPI_Recv(s_current, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, TAG_CUR, MPI_COMM_WORLD, &status);
	}
	else if (status.MPI_TAG == TAG_SUCCESS) {
		MPI_Recv(Success, 1, MPI_INT, MPI_ANY_SOURCE, TAG_SUCCESS, MPI_COMM_WORLD, &status);
	}
	else if (status.MPI_TAG == TAG_PREV) {
		MPI_Recv(s_previous, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, TAG_PREV, MPI_COMM_WORLD, &status);
	}
	else if (status.MPI_TAG == TAG_INITIATIVE) {
		MPI_Recv(ready, 1, MPI_INT, rank - 1, TAG_INITIATIVE, MPI_COMM_WORLD, &status);
	}
	else if (status.MPI_TAG == TAG_STAND) {
		MPI_Recv(standby, 1, MPI_INT, MPI_ANY_SOURCE, TAG_STAND, MPI_COMM_WORLD, &status);
	}
}