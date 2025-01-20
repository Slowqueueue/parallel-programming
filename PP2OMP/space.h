#include "body.h"
#include <omp.h>
#include <vector>

class Space 
{
public:
  Space(){};
  ~Space(){};

  void CalculateTimeNoOMP(unsigned long end_time) 
  {
    while (time_ < end_time) {
      CalcForces();
      time_ += time_dx;
    }
  }

  void CalculateTimeOMP(unsigned long end_time) 
  {
    omp_set_num_threads(omp_get_max_threads());
    while (time_ < end_time) {
      CalcForcesParallel();
      time_ += time_dx;
    }
  }

  void AddBody(Body body) { bodies_.push_back(body); }

  void SetBodies(std::vector<Body> bodies) { bodies_ = bodies; }
  std::vector<Body> GetBodies() { return bodies_; }

private:
  void GravityForce(Body &l_body, Body &r_body) 
  {
    double r = Distance(r_body.GetPosition(), l_body.GetPosition());
    double force = kG_ * (r_body.GetMass() * l_body.GetMass()) / (r * r);
    Vector force_vec_l_body =
        force * Normalize(r_body.GetPosition() - l_body.GetPosition());

    l_body.ApplyForce(force_vec_l_body);
  }

  void Collision(Body &l_body, Body &r_body) 
  {
    double dist = Distance(r_body.GetPosition(), l_body.GetPosition());
    double radius_sum = r_body.GetRadius() + l_body.GetRadius();
    if (dist < radius_sum) 
    {
      double mass = r_body.GetMass() + l_body.GetMass();
      double intersection = radius_sum - dist;

      Vector l_move = intersection *
                      Normalize(l_body.GetPosition() - r_body.GetPosition()) *
                      l_body.GetMass() / mass;

      l_body.SetPosition(l_body.GetPosition() + l_move);

      Vector l_speed = SpeedAfterCollistion(l_body, r_body);
      l_body.SetSpeed(l_speed);
    }
  }

  Vector SpeedAfterCollistion(Body &l_body, Body &r_body) 
  {
    Vector l_speed =
        (2 * r_body.GetMass() * r_body.GetVelocity() -
         (l_body.GetMass() - r_body.GetMass()) * l_body.GetVelocity()) /
        (l_body.GetMass() + r_body.GetMass());
    return l_speed;
  }

  void CalcForces() 
  {
    for (size_t i = 0; i < bodies_.size(); i++) 
    {
      for (size_t j = 0; j < bodies_.size(); j++) 
      {
        if (i == j) continue;
        Collision(bodies_[i], bodies_[j]);
        GravityForce(bodies_[i], bodies_[j]);
      }
    }
    for (size_t i = 0; i < bodies_.size(); i++) 
    {
      bodies_[i].Move();
    }
  }

  void CalcForcesParallel() 
  {
#pragma omp parallel for default(none)
    for (size_t i = 0; i < bodies_.size(); i++) 
    {
      for (size_t j = 0; j < bodies_.size(); j++) 
      {
        if (i == j) continue;
        Collision(bodies_[i], bodies_[j]);
        GravityForce(bodies_[i], bodies_[j]);
      }
    }
    for (size_t i = 0; i < bodies_.size(); i++) 
    {
      bodies_[i].Move();
    }
  }

  unsigned long time_ = 0;
  unsigned long time_dx = 1;
  std::vector<Body> bodies_;
  const double kG_ = 6.6742E-11;
};