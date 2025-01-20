#include <cmath>
using namespace std;
struct Point
{
    float Temp = 0;
    void SetTemperature(double eps, float newTemp, bool* isApprox)
    {
        float OldTemp = Temp;
        Temp = newTemp;
        if (abs(Temp - OldTemp) < eps) {
            *isApprox = true;
        }
    }
};
struct Parallelepiped
{
    int X_len = 0, Y_len = 0, Z_len = 0;
    vector<Point> array;
    Point& GetPoint(int x, int y, int z)
    {
        return array[(x * Y_len + y) * Z_len + z];
    }
    void operator = (const Parallelepiped& p)
    {
        X_len = p.X_len;
        Y_len = p.Y_len;
        Z_len = p.Z_len;
        array = p.array;
    }
    void SetVarTemperature(double T1, double T2)
    {
        GetPoint(X_len / 2, Y_len / 2, Z_len - 1).Temp = T1;
        double t2_shag = T2 / (Z_len - 1);
        for (int z = 1; z < Z_len; z++)
        {
            for (int x = 0; x < X_len; x++)
            {
                GetPoint(x, 0, z).Temp = z * t2_shag;
                GetPoint(x, Y_len - 1, z).Temp = z * t2_shag;
            }
            for (int y = 0; y < Y_len; y++)
            {
                GetPoint(0, y, z).Temp = z * t2_shag;
                GetPoint(X_len - 1, y, z).Temp = z * t2_shag;
            }
        }
    }
    Parallelepiped() {}
    Parallelepiped(int X_len, int Y_len, int Z_len) : X_len(X_len), Y_len(Y_len), Z_len(Z_len)
    {
        array = vector<Point>(X_len * Y_len * Z_len);
    }
};