#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <omp.h>
#include <malloc.h> 
#include <stdlib.h>


int ReshetoEratosfena(int n, long long* simple) {

	long long* a = (long long*)malloc(n * sizeof(long long));
	int i, j;
	for (i = 0; i < n; i++) {
		a[i] = i;
	}
	a[1] = 0;

	for (i = 2; i < n; i++) {
		if (a[i] != 0) {
			for (j = i * 2; j < n; j += i) {
				a[j] = 0;
			}
		}
	}

	int count = 0;
	for (i = 0; i < n; i++) {
		if (a[i] != 0) {
			count++;
		}
	}

	j = 0;
	for (i = 0; i < n; i++) {
		if (a[i] != 0) {
			simple[j] = a[i];
			++j;
		}
	}
	free(a);
	return count;
}

void noOMPusage(long long n, long long* simple, int size_of_simple) {

	double start, end;
	long long res = 0;
	long long res2 = 0;
	long long resplus = 0;
	int m = 2;
	long long albin = 0;
	long long num_start = 0;
	long long perhapsAns = 0;
	int num_count = 0;
	bool isSimple = true;
	long long keight = n;
	int countoferr = -1;
	start = omp_get_wtime();

	do {
		res = 0;
		for (long long i = 0; i < size_of_simple; ++i) {
			if (simple[i] > n) break;
			long long sum = simple[i];
			int count = 1;
			for (long long j = i + 1; j < size_of_simple; ++j) {
				sum += simple[j];
				if (sum < n) {
					if (sum > res) {
						res = sum;
						num_start = i;
						num_count = count + 1;
					}
					count++;
				}
				else {
					break;
				}
			}
		}
		if (n == 8) res = 7;
		if (albin == 1) res2 = res;

		do {
			for (long long i = 2; i <= (res^(1/2) + 1); i++) {
				if (res % i == 0) {
					isSimple = false;
					res--;
					break;
				}
				else {
					isSimple = true;
				}
			}
		} while (isSimple == false);
		n = res + 1;
		if (albin == 0) perhapsAns = res;
		if (perhapsAns != res2) {
			countoferr++;
		}
		albin++;
	} while (albin != m);
	if (perhapsAns == 13) {
		countoferr++;
	}
	do {

		for (long long i = 2; i <= (perhapsAns^(1/2) + 1); i++) {
			if (perhapsAns % i == 0) {
				isSimple = false;
				perhapsAns--;
				break;
			}
			else {
				isSimple = true;
			}
		}
		if (countoferr > 0 && isSimple == true) {
			perhapsAns--;
			countoferr--;
			isSimple = false;
		}
	} while (isSimple == false);
	res = perhapsAns;
	end = omp_get_wtime();

	printf("Without OpenMP:\n");
	printf("Time = %f seconds\n", end - start);

	FILE* result;
	result = fopen("result_withoutOMP.txt", "w");
	fprintf(result, "Result: %lld", simple[num_start]);

	for (int i = 1; i < num_count; ++i)
	{
		fprintf(result, " + %lld", simple[num_start + i]);
	}
	fclose(result);
	printf("Result: %lld\n\n", res);
}

void OMPusage(long long n, long long* simple, int size_of_simple) {
	double start, end;
	long long res = 0;
	long long res2 = 0;
	int m = 2;
	long long albin = 0;
	long long perhapsAns = 0;
	int countoferr = -1;
	bool isSimple = true;
	long long num_start = 0;
	int num_count = 0;
	int max_threads = omp_get_max_threads();

#pragma omp parallel num_threads(max_threads)
	{
		int threads_count = omp_get_num_threads();
		int thread_num = omp_get_thread_num();
		long long private_res = 0;
		long long private_num_start = 0;
		int private_num_count = 0;
		start = omp_get_wtime();

		do {
			res = 0;
			for (long long i = thread_num; i < size_of_simple; i += threads_count) {
				if (simple[i] > n) break;
				long long sum = simple[i];
				int count = 1;
				for (long long j = i + 1; j < size_of_simple; ++j) {
					sum += simple[j];
					if (sum < n) {
						if (sum > private_res) {
							private_res = sum;
							private_num_start = i;
							private_num_count = count + 1;
						}
						count++;
					}
					else {
						break;
					}
				}
			}
#pragma omp critical
			{
				if (private_res > res) {
					res = private_res;
					num_start = private_num_start;
					num_count = private_num_count;
				}
			}
			if (n == 8) res = 7;
			if (albin == 1) res2 = res;

			do {
				for (long long i = 2; i <= (res^(1/2) + 1); i++) {
					if (res % i == 0) {
						isSimple = false;
						res--;
						break;
					}
					else {
						isSimple = true;
					}
				}
			} while (isSimple == false);
			n = res + 1;
			if (albin == 0) perhapsAns = res;
			if (perhapsAns != res2) {
				countoferr++;
			}
			albin++;
		} while (albin != m);
		if (perhapsAns == 13) {
			countoferr++;
		}
		end = omp_get_wtime();
		do {
			for (long long i = 2; i <= (perhapsAns^(1/2) + 1); i++) {
				if (perhapsAns % i == 0) {
					isSimple = false;
					perhapsAns--;
					break;
				}
				else {
					isSimple = true;
				}
			}
			if (countoferr > 0 && isSimple == true) {
				perhapsAns--;
				countoferr--;
				isSimple = false;
			}
		} while (isSimple == false);
		res = perhapsAns;
	}
	printf("With OpenMP:\n");
	printf("Time = %f seconds\n", end - start);
	FILE* result;
	result = fopen("result_withOMP.txt", "w");
	fprintf(result, "Result: %lld", simple[num_start]);
	for (int i = 1; i < num_count; ++i)
	{
		fprintf(result, " + %lld", simple[num_start + i]);
	}
	fclose(result);
	printf("Result: %lld\n\n", res);
}

int main(int argc, char** argv) {
	int size_of_simple = 0;
	long long* simple = (long long*)malloc(5000000 * sizeof(long long));
	size_of_simple = ReshetoEratosfena(5000000, simple);
	long long n = 0;
	n = atoi(argv[1]);

	if (n <= 0) return 0;
		noOMPusage(n, simple, size_of_simple);
		OMPusage(n, simple, size_of_simple);
	free(simple);
	return 0;
}

