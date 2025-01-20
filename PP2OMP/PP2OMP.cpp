/*2. Решение задачи движения N тел на плоскости с учетом сил гравитационного притяжения между ними;
при упругих соударениях количество тел не меняется. Массы, радиусы, начальные координаты и векторы скоростей всех тел
считать заданными. Вычислить координаты заданного тела через k секунд. */
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <string>
#include <ctime>
#include <fstream>
#include "space.h"
using namespace std;

double RandomParameters(double MinBorder, double MaxBorder) //Функция для задания случайных параметров.
{
    double rnd = (double)rand() / RAND_MAX;
    return MinBorder + rnd * (MaxBorder - MinBorder);
}

std::vector<Body> CreateBodies(unsigned N) //Создание тел и присваивание им случайных параметров.
{
    const double MinRadius = 1, MaxRadius = 25;
    const double MinVelocity = 1, MaxVelocity = 1000;
    const double MinPosition = -1000000, MaxPosition = 1000000;
    const double MinMass = 7.35E22, MaxMass = 100000 * 7.35E22;
    std::vector<Body> Bodies;
    for (unsigned i = 0; i < N; i++)
    {
        double Radius = RandomParameters(MinRadius, MaxRadius);
        Vector Velocity(RandomParameters(-1, 1), RandomParameters(-1, 1));
        Velocity = Velocity * RandomParameters(MinVelocity, MaxVelocity);
        double x = RandomParameters(MinPosition, MaxPosition);
        double y = RandomParameters(MinPosition, MaxPosition);
        Vector Position(x, y);
        double Mass = RandomParameters(MinMass, MaxMass);
        Body Body(Position, Velocity, Radius, Mass);
        Bodies.push_back(Body);
    }
    return Bodies;
}

void SaveData(std::vector<Body> state, int B) //Функция для записи данных в выходной файл.
{
    ofstream out("outputfile.txt", std::ios_base::app);
    double Bradius = state[B - 1].GetRadius();
    Vector Bvelocity = state[B - 1].GetVelocity();
    Vector Bposition = state[B - 1].GetPosition();
    double Bmass = state[B - 1].GetMass();
    out << "Number of body: " << B << ". Position of body (x, y) - (" << Bposition.x << ";" << Bposition.y
        << ") Velocity of body (x, y) - (" << Bvelocity.x << ";" << Bvelocity.y << ") Mass of body - "
        << Bmass << " Radius of body - " << Bradius << "\n";
    out << "\n";
    out.close();
}

void StartCalculations(int N, double k, int B) //Функция для запуска вычислений.
{
    Space noOMP, OMP;
    double noOMPtime, OMPtime;
    std::vector<Body> bodies;
    bodies = CreateBodies(N);
    noOMP.SetBodies(bodies);
    OMP.SetBodies(bodies);
    noOMPtime = omp_get_wtime();
    noOMP.CalculateTimeNoOMP(k);
    noOMPtime = omp_get_wtime() - noOMPtime;
    cout << "Time without OMP: " << noOMPtime << " sec.\n";
    OMPtime = omp_get_wtime();
    OMP.CalculateTimeOMP(k);
    OMPtime = omp_get_wtime() - OMPtime;
    cout << "Time with OMP: " << OMPtime << " sec.\n";
    double Diff = noOMPtime - OMPtime;
    for (size_t i = 0; i < noOMP.GetBodies().size(); i++)
        if (noOMP.GetBodies()[i].GetPosition() != OMP.GetBodies()[i].GetPosition())
        {
            cout << "\nError: OMP coordinates and noOMP coordinates are different\n";
            break;
        }
    ofstream out("outputfile.txt", std::ios_base::app);
    out << "[Raw data]: N - " << bodies.size() << ", k - " << k << " sec, B - " << B << "\n";
    out.close();
    ofstream out2("outputfile.txt", std::ios_base::app);
    out2 << "[Result data]\nTime without OMP: " << noOMPtime
        << " sec.\nTime with OMP: " << OMPtime << " sec.\n"
        << (Diff > 0 ? "Version with OMP" : "Version without OMP") << " is faster by "
        << (Diff > 0 ? Diff : -Diff) << " sec.\n";
    out2.close();
    SaveData(OMP.GetBodies(), B);
    cout << "Data saved in outputfile.txt\n";
}

int main(int argc, char** argv) //Функция main, аргументы: .exe, number of bodies, time, specified body.
{
    srand(time(0));
    ofstream out("outputfile.txt", std::ios_base::trunc);
    out.close();
    int N = atoi(argv[1]);
    double k = atoi(argv[2]);
    int B = atoi(argv[3]);
    StartCalculations(N, k, B);
    return 0;
}