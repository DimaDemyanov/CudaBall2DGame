#pragma once
#include"Vector.h"
#include"Drawable.h"
#include"Drawer.h"
#include"Basket.h"
#include"Platform.h"
#include"CollisionChecker.h"
#include <gl/glut.h>
#include <list>

enum BallType {
  RED, BLUE
};

const Color RED_BALL_COLOR = { 1.0f, 0.0f, 0.0f };
const Color BLUE_BALL_COLOR = { 0.0f, 0.0f, 1.0f };

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
    this->mass = radius * radius * 100000;
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
  static void move(std::list<Ball*> balls, int deltaTime, std::list<Platform*> platforms) {
    BallData* ballsData = convertBallsListToBallsData(balls);
    CudaMoveBalls(ballsData, balls.size(), 1.0 * deltaTime / 3000, Platform::convertPlatformsListToPlatformsData(platforms), platforms.size());
    convertBallsDataToBallsList(ballsData, balls.size(), balls);
  }

  static void Ball::CudaMoveBalls(BallData* ballsData, int ballsCount, float deltaTime, PlatformData* platformData, int platformDataCount);// {};

  bool checkCollisionWithBasket(Basket* basket) {
    return CollisionChecker::intersectsBallAndRect(pos->getX(), pos->getY(), radius, basket->getX1(), basket->getY2(), basket->getX2(), basket->getY1());
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

  void setBallData(BallData ballData) {
    pos->setX(ballData.pos_x);
    pos->setY(ballData.pos_y);
    velocity->setX(ballData.velocity_x);
    velocity->setY(ballData.velocity_y);
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

  static void convertBallsDataToBallsList(BallData* ballsData, int size, std::list<Ball*> balls) {
    int i = 0;
    for (Ball* ball: balls) {

      ball->setBallData(ballsData[i++]);
    }
  }
};