#include <cmath>
#include <limits>

struct Vector 
{
  double x = 0;
  double y = 0;

  Vector(double x_, double y_) 
  {
    x = x_;
    y = y_;
  }

  struct Vector &operator+=(const Vector &rhs) 
  {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }
  struct Vector &operator-=(const Vector &rhs) 
  {
    x -= rhs.x;
    y -= rhs.y;
    return *this;
  }
  struct Vector &operator+=(const double &k) 
  {
    x += k;
    y += k;
    return *this;
  }
  struct Vector &operator*=(const double &k) 
  {
    x *= k;
    y *= k;
    return *this;
  }
  struct Vector &operator/=(const double &k) 
  {
    x /= k;
    y /= k;
    return *this;
  }
};

Vector operator+(Vector lhs, const Vector &rhs) { return lhs += rhs; }
Vector operator+(Vector lhs, const double k) { return lhs += k; }
Vector operator+(const double k, Vector rhs) { return rhs += k; }
Vector operator-(Vector lhs, const Vector &rhs) { return lhs -= rhs; }
Vector operator*(Vector lhs, const double k) { return lhs *= k; }
Vector operator*(const double k, Vector rhs) { return rhs *= k; }
Vector operator/(Vector lhs, const double k) { return lhs /= k; }

bool operator==(Vector lhs, Vector rhs) 
{
  return (fabs(lhs.x - rhs.x) < std::numeric_limits<double>::epsilon() &&
          fabs(lhs.y - rhs.y) < std::numeric_limits<double>::epsilon());
}

bool operator!=(Vector lhs, Vector rhs) { return !(lhs == rhs); }

double Distance(Vector p1, Vector p2) 
{
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

double Length(Vector point) { return Distance(point, {0, 0}); }
Vector Normalize(Vector point) { return point / Length(point); }
