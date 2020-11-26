#pragma once 
#include"Vector.h"

const float ANGEL_DELTA = 0.5f;

class Platform: public Drawable {
public:
  Platform(Vector* pos, float angle, float width, float height) {
    this->pos = new Vector(*pos);
    this->angle = angle;
    this->width = width;
    this->height = height;
  }

  void rotateRight() {
    angle -= ANGEL_DELTA;
  }

  void rotateLeft() {
    angle += ANGEL_DELTA;
  }

  void draw() {
    glPushMatrix();
    glTranslatef(pos->getX(), pos->getY(), 0);
    glRotatef(angle, 0, 0, 1);
    glBegin(GL_QUADS);
    glVertex2f(-width / 2.0f, -height / 2.0f);
    glVertex2f(width / 2.0f, -height / 2.0f);
    glVertex2f(width / 2.0f, height / 2.0f);
    glVertex2f(-width / 2.0f, height / 2.0f);
    glEnd();
    glPopMatrix();
  }

  Vector* getPos() {
    return pos;
  }

  void setPos(Vector* pos) {
    this->pos = new Vector(*pos);
  }

  float getAngle() {
    return angle;
  }

  void setAngle(float angle) {
    this->angle = angle;
  }

  float getWidth() {
    return width;
  }

  void setWidth(float width) {
    this->width = width;
  }

  float getHeight() {
    return height;
  }

  void setHeight(float height) {
    this->height = height;
  }

private:
  Vector* pos;
  float angle;
  float width;
  float height;
};