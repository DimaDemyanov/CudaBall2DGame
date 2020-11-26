#pragma once
#include"Vector.h"
#include"Drawable.h"
#include"Drawer.h"
#include"Basket.h"
#include"Platform.h"
#include"CollisionChecker.h"
#include <gl/glut.h>

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

  void move(float deltaTime) {
    pos->setX(pos->getX() - velocity->getX() * deltaTime);
    pos->setY(pos->getY() - velocity->getY() * deltaTime);
    velocity->setY(velocity->getY() + g * deltaTime);
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

  bool colliding(Ball* ball);

  void resolveCollision(Ball* ball);
 

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
};