#include "vector.h"
class Body 
{
public:
  Body(Vector pos, Vector move, double radius, double mass)
      : position_(pos), velocity_(move), radius_(radius), mass_(mass) {}
  void Move() { position_ = position_ + velocity_; }
  void ApplyForce(Vector force) { velocity_ += force / mass_; }
  void SetSpeed(Vector move) { velocity_ = move; }
  Vector SetPosition(Vector p) { return position_ = p; }

  double GetMass() { return mass_; }
  double GetRadius() { return radius_; }
  Vector GetPosition() { return position_; }
  Vector GetVelocity() { return velocity_; }
  void PrintVec() 
  {
    std::cout << position_.x << " " << position_.y;
  }
private:
  Vector position_;
  Vector velocity_;
  double radius_;
  double mass_;
};
