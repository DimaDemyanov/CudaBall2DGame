#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"Ball.h"

bool Ball::colliding(Ball* ball)
{
  float xd = pos->getX() - ball->getPos()->getX();
  float yd = pos->getY() - ball->getPos()->getY();

  float sumRadius = getRadius() + ball->getRadius();
  float sqrRadius = sumRadius * sumRadius;

  float distSqr = (xd * xd) + (yd * yd);

  if (distSqr <= sqrRadius)
  {
    return true;
  }

  return false;
}

void Ball::resolveCollision(Ball* ball)
{
  // get the mtd
  Vector* delta = pos->subtract(ball->pos);
  float d = delta->getLength();
  // minimum translation distance to push balls apart after intersecting
  Vector* mtd = delta->multiply(((getRadius() + ball->getRadius()) - d) / d);


  // resolve intersection --
  // inverse mass quantities
  float im1 = 1 / getMass();
  float im2 = 1 / ball->getMass();

  // push-pull them apart based off their mass
  //pos = pos->add(mtd->multiply(im1 / (im1 + im2)));
  //ball->pos = ball->getPos()->subtract(mtd->multiply(im2 / (im1 + im2)));

  // impact speed
  Vector* v = velocity->subtract(ball->velocity);
  float vn = v->dot(mtd->normalize());

  // sphere intersecting but moving away from each other already
  if (vn < 0.0f) {
    return;
  }

  // collision impulse
  float i = (-(1.0f + 1.0f) * vn) / (im1 + im2);
  Vector* impulse = mtd->normalize()->multiply(i);

  // change in momentum
  velocity = velocity->add(impulse->multiply(im1));
  ball->setVelocity(ball->velocity->subtract(impulse->multiply(im2)));

}