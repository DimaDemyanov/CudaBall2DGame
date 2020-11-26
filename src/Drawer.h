#pragma once
//#include <Windows.h>
#include <GL/glut.h>
#include <math.h>

const float M_PI = 3.14159265f;

struct Color {
  float r, g, b;
};

class Drawer {
public:
  static void drawCircle(float x, float y, float radius) {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(x, y, 0.0f);
    static const int circle_points = 100;
    static const float angle = 2.0f * 3.1416f / circle_points;

    glBegin(GL_POLYGON);
    double angle1 = 0.0;
    glVertex2d(radius * cos(0.0), radius * sin(0.0));
    int i;
    for (i = 0; i < circle_points; i++)
    {
      glVertex2d(radius * cos(angle1), radius *sin(angle1));
      angle1 += angle;
    }
    glEnd();
    glPopMatrix();
  }

  static void setColor(Color color) {
    glColor3f(color.r, color.g, color.b);
  }
};