#pragma once

#include<cmath>

class Vector {
public: 
  Vector(float x, float y) {
    this->x = x;
    this->y = y;
  }

  Vector(const Vector &v) {
    x = v.x;
    y = v.y;
  }

  float getX() {
    return x;
  }

  void setX(float x) {
    this->x = x;
  }

  float getY() {
    return y;
  }

  void setY(float y) {
    this->y = y;
  }

  Vector* add(Vector* v) {
    return new Vector(x + v->getX(), y + v->getY());
  }

  Vector* subtract(Vector* v) {
    return new Vector(x - v->getX(), y - v->getY());
  }

  Vector* multiply(float k) {
    return new Vector(x * k, y * k);
  }

  float getLength() {
    return sqrt(x * x + y * y);
  }

  Vector* normalize() {
    float length = getLength();
    return new Vector(x / length, y / length);
  }

  float dot(Vector* v) {
    return x * v->getX() + y * v->getY();
  }
private:
  float x, y;
};