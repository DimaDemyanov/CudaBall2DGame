#pragma once
#include"Vector.h"
#include"Ball.h"
#include <random>
#include <ctime>

class BallGenerator {
public:
  BallGenerator(Vector* velocityFrom, Vector* velocityTo, Vector* generatorPos, float ballsRadius, BallType ballsType) {
    std::srand(std::time(nullptr));
    this->velocityFrom = velocityFrom;
    this->velocityTo = velocityTo;
    this->generatorPos = generatorPos;
    this->ballsRadius = ballsRadius;
    this->ballsType = ballsType;
  }

  Ball* generateBall() {
    double r1 = ((double)rand() / (RAND_MAX + 1));
    double r2 = ((double)rand() / (RAND_MAX + 1));
    Vector* ballPos = new Vector(*generatorPos);
    Vector* ballVelocity = new Vector(velocityFrom->getX() + (velocityTo->getX() - velocityFrom->getX()) * r1, velocityFrom->getY() + (velocityTo->getY() - velocityFrom->getY()) * r2);
    Ball* ball = new Ball(ballPos, ballVelocity, ballsRadius, ballsType);
    return ball;
  }
private:
  Vector* velocityFrom;
  Vector* velocityTo;
  Vector* generatorPos;
  float ballsRadius;
  BallType ballsType;
};