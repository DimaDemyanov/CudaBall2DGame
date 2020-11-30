#pragma once
#include"Drawable.h"
#include"Drawer.h"
#include"Platform.h"
#include <gl/glut.h>

const Color BASKET_COLOR = {0.4f, 0.4f, 0.4f};

class Basket: public Drawable {
public:
  Basket(float length, float height) {
    this->length = length;
    this->height = height;
  }

  void draw() {
    Drawer::setColor(BASKET_COLOR);
    glRectf(getX1(), getY1(), getX2(), getY2());
  }

  float getX1() {
    return -length / 2;
  }

  float getY1() {
    return -1;
  }

  float getX2() {
    return length / 2;
  }

  float getY2() {
    return -1 + height;
  }

  float getLength() {
    return length;
  }

  void setLength(float length) {
    this->length = length;
  }

  float getHeight() {
    return height;
  }

  void setHeight(float height) {
    this->height = height;
  }

  PlatformData getPlatformData() {
    PlatformData platformData = { 0.0f, -1 + height / 2, 0.0f, length, height };
    return platformData;
  }

private:
  float length;
  float height;
};