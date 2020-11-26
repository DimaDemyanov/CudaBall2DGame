#include "cuda_runtime.h"
#include "device_launch_parameters.h"

bool collidingGPU(Ball* ball)
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