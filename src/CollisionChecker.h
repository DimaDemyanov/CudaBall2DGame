#pragma once
#include<cmath>
#include "Vector.h"

#define PI 3.14159265

class CollisionChecker {
public: 
  static bool intersectsBallAndRect(float cx, float cy, float radius, float left, float top, float right, float bottom) {
    float closestX = (cx < left ? left : (cx > right ? right : cx));
    float closestY = (cy > top ? top : (cy < bottom ? bottom : cy));
    float dx = closestX - cx;
    float dy = closestY - cy;

    return (dx * dx + dy * dy) <= radius * radius;
  }

  static Vector* getIntersectionNormal(float cx, float cy, float radius, float left, float top, float right, float bottom) {
    float closestX = (cx < left ? left : (cx > right ? right : cx));
    float closestY = (cy > top ? top : (cy < bottom ? bottom : cy));
    float dx = closestX - cx;
    float dy = closestY - cy;

    Vector* normal = new Vector(0.0f, 0.0f);

    if (dy > dx && dx != 0 ) {
      if (cx < left) {
        normal->setX(-1.0f);
      }
      else {
        normal->setX(1.0f);
      }
    }
    else {
      if (cy < top) {
        normal->setY(1.0f);
      }
      else {
        normal->setY(-1.0f);
      }
    }

    return normal;
  }

  static Vector* changeVelocity(float cx, float cy, float radius, float left, float top, float right, float bottom, Vector* velocity) {
    float closestX = (cx < left ? left : (cx > right ? right : cx));
    float closestY = (cy < top ? top : (cy > bottom ? bottom : cy));
    float dx = closestX - cx;
    float dy = closestY - cy;

    Vector* newVelocity = new Vector(*velocity);
    if (dy > dx && dx != 0) {
      if (dx * velocity->getX() > 0) {
        return new Vector(*velocity);
      }
      newVelocity->setX(-velocity->getX());
    }
    else {
      if (dy * velocity->getY() > 0) {
        return new Vector(*velocity);
      }
      newVelocity->setY(-velocity->getY());
    }

    return newVelocity;
  }

  static bool intersectsBallAndRotatedRect(float cx, float cy, float radius, float x, float y, float width, float height, float angle) {
    float alpha = - angle / 180 * PI;
    float x1 = (cx - x) * cos(alpha) - (cy - y) * sin(alpha);
    float y1 = (cx - x) * sin(alpha) + (cy - y) * cos(alpha);
    return intersectsBallAndRect(x1, y1, radius, -width / 2, height / 2, width / 2, -height / 2);
  }

  static Vector* changeVelocity(float cx, float cy, float radius, float x, float y, float width, float height, float angle, Vector* velocity) {
    float alpha = - angle / 180 * PI;
    float x1 = (cx - x) * cos(alpha) - (cy - y) * sin(alpha);
    float y1 = (cx - x) * sin(alpha) + (cy - y) * cos(alpha);
    Vector* newVelocity = new Vector(velocity->getX() * cos(alpha) - velocity->getY() * sin(alpha), velocity->getX() * sin(alpha) + velocity->getY() * cos(alpha));
    newVelocity =  changeVelocity(x1, y1, radius, -width / 2, height / 2, width / 2, -height / 2, newVelocity);
    return new Vector(newVelocity->getX() * cos(alpha) + newVelocity->getY() * sin(alpha), -newVelocity->getX() * sin(alpha) + newVelocity->getY() * cos(alpha));
  }
};