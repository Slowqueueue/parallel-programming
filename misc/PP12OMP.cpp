#include <iostream>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <map>
#include <time.h>
using namespace std;

int c_v;
int** MST;
int** MST_omp;
vector<int> vs;
vector<int> included_vertexes;

void randomize_graph(int c);
void findseq();
void findpar();

int main(int argc, char* argv[])
{
	ofstream out("output.txt");
	double start, end;
	if (argc != 2) {
		printf("Wrong arguments. Arguments: number of vertexes");
		return 0;
	}

	int c = atoi(argv[1]);
	randomize_graph(c);

	start = omp_get_wtime();
	findseq();
	end = omp_get_wtime();
	cout << "Sequential time: " << end - start << endl;

	start = omp_get_wtime();
	findpar();
	end = omp_get_wtime();
	cout << "Parallel time: " << end - start << endl;

	for (int i = 0; i < c_v; i++)
		for (int j = 0; j < c_v; j++)
			if (MST[i][j] != MST_omp[i][j]) {
				cout << "Results are different" << endl;
				return 0;
			}
	cout << "Results are equal" << endl;
	for (int i = 0; i < c_v; i++)
		for (int j = 0; j < c_v; j++)
			out << MST[i][j] << " ";
	out.close();
	return 0;
}

void initialMst(int c)
{
	c_v = c;
	MST = new int* [c];
	MST_omp = new int* [c];
	for (int i = 0; i < c; i++)
	{
		MST[i] = new int[c];
		MST_omp[i] = new int[c];
	}

	for (int i = 0; i < c_v; i++)
		for (int j = 0; j < c_v; j++)
		{
			MST[i][j] = 0;
			MST_omp[i][j] = 0;
		}
}

void taskclear()
{
	vs.clear();
	for (int i = 0; i < c_v; i++)
		vs.push_back(i);
	included_vertexes.clear();
}

bool searchInIncluded(int v)
{
	int l = included_vertexes.size();
	for (int i = 0; i < l; i++)
		if (included_vertexes.at(i) == v) return true;
	return false;
}

struct verge
{
	int dist_vertex;
	int from_vertex;
	int weight;
};

struct min_edge
{
	verge e;
	int index_vertex;
};

map<int, vector<verge> > graph;

void addVerge(int v1, int v2, int weight)
{
	verge e;
	e.dist_vertex = v2;
	e.from_vertex = v1;
	e.weight = weight;
	graph[v1].push_back(e);
	e.dist_vertex = v1;
	e.from_vertex = v2;
	graph[v2].push_back(e);
}

void addVertex(int v)
{
	vector<verge> l;
	graph.insert(pair<int, vector<verge> >(v, l));
}

void randomize_graph(int c)
{
	initialMst(c);
	for (int i = 0; i < c; i++)
		addVertex(i);
	float w;
	int r2 = RAND_MAX / 2;
	for (int i = 0; i < c; i++)
		for (int j = i + 1; j < c; j++)
		{
			w = rand() * 0.99;
			addVerge(i, j, w);
		}
}

bool isIdentical()
{
	for (int i = 0; i < c_v; i++)
		for (int j = 0; j < c_v; j++)
			if (MST[i][j] != MST_omp[i][j])
				return false;
	return true;
}

min_edge minVergeSeq()
{
	int s = vs.size();
	int mi;
	int mw = RAND_MAX;
	min_edge eee;
	verge ee;
	eee.index_vertex = -1;
	for (int i = 0; i < s; i++)
	{
		int v = vs.at(i);
		int sv = graph[v].size();
		int lmw = RAND_MAX;
		bool lmwf = false;
		verge lme;
		for (int j = 0; j < sv; j++)
		{
			verge e = graph[v].at(j);
			if (searchInIncluded(e.dist_vertex))
				if (e.weight < lmw)
				{
					lmwf = true;
					lmw = e.weight;
					lme = e;

				}
		}
		if (!lmwf) continue;
		if (lmw < mw)
		{
			ee = lme;
			mi = i;
			mw = lmw;
		}
	}
	eee.index_vertex = mi;
	eee.e = ee;
	return eee;
}

min_edge minVergePar()
{
	int s = vs.size();
	int mi;
	int mw = RAND_MAX;
	min_edge ee;
	verge e;
	ee.index_vertex = -1;
#pragma omp parallel for default(none) shared(e,graph,mi, mw, vs, s)
	for (int i = 0; i < s; i++)
	{
		int v = vs.at(i);
		int sv = graph[v].size();
		int lmw = RAND_MAX;
		bool lmwf = false;
		verge lme;
		for (int j = 0; j < sv; j++)
		{
			verge e = graph[v].at(j);
			if (searchInIncluded(e.dist_vertex))
				if (e.weight < lmw)
				{
					lmwf = true;
					lmw = e.weight;
					lme = e;

				}
		}
		if (!lmwf) continue;
#pragma omp critical
		if (lmw < mw)
		{
			e = lme;
			mi = i;
			mw = lmw;
		}
	}
	ee.index_vertex = mi;
	ee.e = e;
	return ee;
}

void findseq()
{
	taskclear();
	included_vertexes.push_back(vs.at(0));
	vs.erase(vs.begin());
	while (vs.size() > 0)
	{
		min_edge e = minVergeSeq();
		if (e.index_vertex > -1)
		{
			included_vertexes.push_back(e.e.from_vertex);
			vs.erase(vs.begin() + e.index_vertex);
			MST[e.e.dist_vertex][e.e.from_vertex] = e.e.weight;
			MST[e.e.from_vertex][e.e.dist_vertex] = e.e.weight;
		}
	}
}

void findpar()
{
	taskclear();
	included_vertexes.push_back(vs.at(0));
	vs.erase(vs.begin());
	while (vs.size() > 0)
	{
		min_edge e = minVergePar();
		if (e.index_vertex > -1)
		{
			included_vertexes.push_back(e.e.from_vertex);
			vs.erase(vs.begin() + e.index_vertex);
			MST_omp[e.e.dist_vertex][e.e.from_vertex] = e.e.weight;
			MST_omp[e.e.from_vertex][e.e.dist_vertex] = e.e.weight;
		}
	}
}