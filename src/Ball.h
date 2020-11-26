#pragma once
#include"Vector.h"
#include"Drawable.h"
#include"Drawer.h"
#include"Basket.h"
#include"Platform.h"
#include"CollisionChecker.h"
//#include"CudaBall.h"
#include <gl/glut.h>
#include <list>

enum BallType {
  RED, BLUE
};

const Color RED_BALL_COLOR = { 1.0f, 0.0f, 0.0f };
const Color BLUE_BALL_COLOR = { 0.0f, 0.0f, 1.0f };
const float g = 0.2f;

struct BallData {
  float pos_x;
  float pos_y;
  float velocity_x;
  float velocity_y;
  float radius;
  float mass;
  BallType type;
};



class Ball: public Drawable {
public: 
  Ball(Vector* pos, Vector* velocity, float radius, BallType ballType) {
    mass = 1;
    this->pos = new Vector(*pos);
    this->initPos = new Vector(*pos);
    this->velocity = new Vector(*velocity);
    this->radius = radius;
    this->ballType = ballType;
  }

  ~Ball() {
    delete pos;
    delete initPos;
    delete velocity;
  }

  void draw() {
    if (ballType == RED) {
      Drawer::setColor(RED_BALL_COLOR);
    }
    else {
      Drawer::setColor(BLUE_BALL_COLOR);
    }
    Drawer::drawCircle(pos->getX(), pos->getY(), radius);
  }

  bool shouldBeDeleted() {
    if (pos->getY() < -1.5f || pos->getY() > 1.5f || pos->getX() < -1.5f || pos->getX() > 1.5f) {
      return true;
    }
    return false;
  }

  //TODO: avoid convertions to BallData and back
  static std::list<Ball*> move(std::list<Ball*> balls, int deltaTime) {
    BallData* ballsData = convertBallsListToBallsData(balls);
    CudaMoveBalls(ballsData, balls.size(), deltaTime);
    return convertBallsDataToBallsList(ballsData, balls.size());
  }

  static void CudaMoveBalls(BallData* ballsData, int size, int deltaTime);// {};

  //static void move(std::list<Ball*> balls, int deltatime) {
  //  for (Ball* ball : balls) {
  //    ball->move(1.0 * deltatime / 1000);
  //  }
  //}

  void move(float deltatime) {
    pos->setX(pos->getX() - velocity->getX() * deltatime);
    pos->setY(pos->getY() - velocity->getY() * deltatime);
    velocity->setY(velocity->getY() + g * deltatime);
  }

  bool checkCollisionWithBasket(Basket* basket) {
    return CollisionChecker::intersectsBallAndRect(pos->getX(), pos->getY(), radius, basket->getX1(), basket->getY2(), basket->getX2(), basket->getY1());
  }

  void changeVelocity(Basket* basket) {
    velocity = CollisionChecker::changeVelocity(pos->getX(), pos->getY(), radius, basket->getX1(), basket->getY2(), basket->getX2(), basket->getY1(), velocity);
  }

  bool checkCollisionWithPlatform(Platform* platform) {
    return CollisionChecker::intersectsBallAndRotatedRect(pos->getX(), pos->getY(), radius, platform->getPos()->getX(), platform->getPos()->getY(), platform->getWidth(), platform->getHeight(), platform->getAngle());
  }

  void changeVelocity(Platform* platform) {
    velocity = CollisionChecker::changeVelocity(pos->getX(), pos->getY(), radius, platform->getPos()->getX(), platform->getPos()->getY(), platform->getWidth(), platform->getHeight(), platform->getAngle(), velocity);
  }

  bool CudaColliding(Ball* ball1, Ball * ball2);// { return true; }

  bool colliding(Ball* ball) {
    return CudaColliding(this, ball);
  }

  void CudaResolveCollision(Ball* ball1, Ball * ball2);// {}

  void resolveCollision(Ball* ball) {
    return CudaResolveCollision(this, ball);
  }
 

  Vector* getPos() {
    return pos;
  }

  void setPos(Vector* pos) {
    this->pos = pos;
  }

  Vector* getInitPos() {
    return initPos;
  }

  void setInitPos(Vector* initPos) {
    this->initPos = initPos;
  }

  Vector* getVelocity() {
    return velocity;
  }

  void setVelocity(Vector* velocity) {
    this->velocity = velocity;
  }

  float getRadius() {
    return radius;
  }
  
  void setRadius(float radius) {
    this->radius = radius;
  }

  BallType getBallType() {
    return ballType;
  }

  void setBallType(BallType ballType) {
    this->ballType = ballType;
  }

  float getMass() {
    return mass;
  }

  void setMass(float mass) {
    this->mass = mass;
  }

  BallData getBallData() {
    BallData ballData = { pos->getX(), pos->getY(), velocity->getX(), velocity->getY(), radius, mass, ballType };
    return ballData;
  }
private: 
  Vector* pos;
  Vector* initPos;
  Vector* velocity;
  float radius;
  float mass;
  BallType ballType;

  static BallData* convertBallsListToBallsData(std::list<Ball*> balls) {
    BallData* ballsData = new BallData[balls.size()];
    int i = 0;
    for (Ball* ball : balls) {
      ballsData[i++] = ball->getBallData();
    }

    return ballsData;
  }

  static std::list<Ball*> convertBallsDataToBallsList(BallData* ballsData, int size) {
    std::list<Ball*> balls;

    for (int i = 0; i < size; i++) {
      Ball* ball = new Ball(new Vector(ballsData[i].pos_x, ballsData[i].pos_y), new Vector(ballsData[i].velocity_x, ballsData[i].velocity_y), ballsData[i].radius, ballsData[i].type);
      balls.push_back(ball);
    }

    return balls;
  }
};