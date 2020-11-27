#pragma once 
#include"Vector.h"
#include<list>

const float ANGEL_DELTA = 0.5f;

struct PlatformData {
  float pos_x;
  float pos_y;
  float angle;
  float width;
  float height;
};

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

  void moveTo(float d_x, float d_y) {
    pos->setX(pos->getX() + d_x);
    pos->setY(pos->getY() + d_y);
  }
  
  PlatformData getPlatformData() {
    PlatformData platformData = { pos->getX(), pos->getY(), angle, width, height };
    return platformData;
  }

  static PlatformData* convertPlatformsListToPlatformsData(std::list<Platform*> platforms) {
    PlatformData* platformsData = new PlatformData[platforms.size()];
    int i = 0;
    for (Platform* platform : platforms) {
      platformsData[i++] = platform->getPlatformData();
    }

    return platformsData;
  }
private:
  Vector* pos;
  float angle;
  float width;
  float height;
};