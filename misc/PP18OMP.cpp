#include <cstdlib>
#include <time.h>
#include <vector>
#include <omp.h>
#include <iostream>
#include <fstream>
using namespace std;

#define ziffer(v, shift) (((v) >> shift) & ((1 << 8) - 1))

vector<unsigned> SequentialMethod(vector<unsigned> array)
{
	double start, end;
	start = omp_get_wtime();

	int unsSize = sizeof(unsigned) * 8;
	vector<unsigned> buff(array.size());
	for (int step = 0; step < unsSize; step += 8) {
		vector<unsigned> count1((1 << 8), 0);
		vector<unsigned> count2((1 << 8), 0);
		for (int i = 0; i < array.size(); i++) {
			int pos = ziffer(array[i], step);
			count2[pos]++;
		}
		for (int i = 0; i < (1 << 8); i++) count1[i] += count2[i];
		for (int i = 1; i < (1 << 8); i++) count1[i] += count1[i - 1];
		for (int i = 0; i < (1 << 8); i++) {
			count1[i] -= count2[i];
			count2[i] = count1[i];
		}
		for (int i = 0; i < array.size(); i++) {
			int pos = ziffer(array[i], step);
			buff[count2[pos]++] = array[i];
		}
		vector<unsigned> temp = array;
		array = buff;
		buff = temp;
	}

	end = omp_get_wtime();
	printf("\nSequential time: %.7f s\n", (end - start));
	return array;
}

vector<unsigned> ParallelMethod(vector<unsigned> array)
{
	double start, end;
	int unsSize = sizeof(unsigned) * 8;
	vector<unsigned> buff(array.size());
	start = omp_get_wtime();

	for (int step = 0; step < unsSize; step += 8) {
		vector<unsigned> count1((1 << 8), 0);
		vector<unsigned> count2((1 << 8), 0);
#pragma omp parallel firstprivate(counter2)
		{
#pragma omp for schedule(static) nowait
			for (int i = 0; i < array.size(); i++)
			{
				int pos = ziffer(array[i], step);
				count2[pos]++;
			}
#pragma omp critical
			for (int i = 0; i < (1 << 8); i++) count1[i] += count2[i];
#pragma omp barrier
#pragma omp single
			for (int i = 1; i < (1 << 8); i++) count1[i] += count1[i - 1];
			int nofthreads = omp_get_num_threads();
			int thrnum = omp_get_thread_num();
			for (int curT = nofthreads - 1; curT >= 0; curT--)
			{
				if (curT == thrnum)
					for (int i = 0; i < (1 << 8); i++)
					{
						count1[i] -= count2[i];
						count2[i] = count1[i];
					}
				else { 
#pragma omp barrier
				}
			}
#pragma omp for schedule(static)
			for (int i = 0; i < array.size(); i++)
			{ 
				int pos = ziffer(array[i], step);
				buff[count2[pos]++] = array[i];
			}
		}
		vector<unsigned> tmp = array;
		array = buff;
		buff = tmp;
	}

	end = omp_get_wtime();
	printf("Parallel time: %.7f s\n", (end - start));
	return array;
}

int main(int argc, char** argv) {
	srand(time(NULL));
	ofstream fout("output.txt", ios_base::trunc);
	int size;
	bool isIdent = true, isSort = true;
	vector<unsigned> A(0);
	if (argc != 2) {
		cout << "Error. Wrong arguments. Arguments: size of randomized array\n" << endl;
		return 0;
	}
	size = atoi(argv[1]);
	if (size <= 0) {
		cout << "\nSize cannot be negative!\n";
		return 0;
	}
	for (size_t i = 0; i < size; ++i)
		A.push_back(rand() % 10000);
	vector<unsigned> B = vector<unsigned>(A);
	vector<unsigned> init = vector<unsigned>(B);
	A = SequentialMethod(A);
	B = ParallelMethod(B);
	for (int i = 0; i < size; i++)
		if (A[i] != B[i]) isIdent = false;
	cout << (isIdent == true ? "Arrays are identical\n" : "Arrays are not identical\n");
	for (size_t i = 1; i < A.size(); i++)
		if (A[i] < A[i - 1]) isSort = false;
	cout << (isSort == true ? "Arrays are sorted correctly\n" : "Arrays are not sorted correctly\n");
	for (int i = 0; i < size; i++) fout << B[i] << " ";
	fout.close();
	cout << "Sorted array is saved in output.txt\n";
	return 0;
}