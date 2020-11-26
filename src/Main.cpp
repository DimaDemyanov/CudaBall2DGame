#include <windows.h>  // For MS Windows
#include "Game.h"
#include "Ball.h"
#include "Vector.h"
#include <gl/glut.h>  // GLUT, includes glu.h and gl.h
#define WIN32_LEAN_AND_MEAN
#define GLEW_STATIC
#define WIDTH 800
#define HEIGHT 800

Game game;

void display() {
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
  glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer

  game.move();
  game.draw();

  glFlush();  // Render now
  glutSwapBuffers();
}

void keyPressed(unsigned char key, int x, int y) {
  game.handleKey(key);
}

/* Main function: GLUT runs as a console application starting at main()  */
int main(int argc, char** argv) {
  game = Game();
  glutInit(&argc, argv);                 // Initialize GLUT
  glutInitWindowSize(1000, 1000);   // Set the window's initial width & height
  glutInitWindowPosition(0, 0); // Position the window's initial top-left corner
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutCreateWindow("Balls game"); // Create a window with the given title
  glutKeyboardFunc(keyPressed);
  GLenum err = glewInit();
  if (GLEW_OK != err)
  {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }
  glutDisplayFunc(display); // Register display callback handler for window re-paint
  glutIdleFunc(display);
  glutMainLoop();           // Enter the infinitely event-processing loop
  return 0;
}