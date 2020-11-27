#pragma once
#include <list>
#include <iostream>
#define GLT_IMPLEMENTATION
#include <gl/glew.h>
#include <gl/gl.h>
#include <gl/glut.h>
#include"Ball.h"
#include"BallGenerator.h"
#include"Basket.h"
#include"Platform.h"

const unsigned int TIME_BETWEEN_BALLS_GENERATION = 10;
const unsigned int BALLS_MAX_COUNT = 3000;
const unsigned int RED_BALLS_COUNT_TO_WIN = 70;
const unsigned int BLUE_BALLS_COUNT_TO_WIN = 60;

class Game {
public:
  Game() {
    initGame();
  }

  void initGame() {
    //font = gltext::Font("C:/Users/dimad/source/repos/CudaKursachCMakeV2.0/resources/Lato-Thin.ttf", 16, 128);
    //font.setDisplaySize(300, 300);
    //font.cacheCharacters("1234567890!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,./;'[]\\<>?:\"{}|-=_+");
    redCount = blueCount = 0;
    oldTimeSinceStart = glutGet(GLUT_ELAPSED_TIME);
    Vector* redBallsGenPos = new Vector(1, 1);
    Vector* redBallsGenVelocityFrom = new Vector(0.1, 0.1);
    Vector* redBallsGenVelocityTo = new Vector(0.7, 0.2);
    Vector* blueBallsGenPos = new Vector(-1, 1);
    Vector* blueBallsGenVelocityFrom = new Vector(-0.1, 0.1);
    Vector* blueBallsGenVelocityTo = new Vector(-0.7, 0.2);
    Vector* velocity = new Vector(0.1, 0.1);

    Vector* platform1Pos = new Vector(0.5, 0.5);
    Vector* platform2Pos = new Vector(-0.6, -0.2);
    Vector* platform3Pos = new Vector(0, 0);

    platform1 = new Platform(platform1Pos, 0.0f, 0.4f, 0.05f);
    platform2 = new Platform(platform2Pos, 0.0f, 0.45f, 0.07f);
    platform3 = new Platform(platform3Pos, 45.0f, 0.2f, 0.2f);

    platforms.push_back(platform1);
    platforms.push_back(platform2);
    platforms.push_back(platform3);

    redBallGenerator = new BallGenerator(redBallsGenVelocityFrom, redBallsGenVelocityTo, redBallsGenPos, 0.015, RED);
    blueBallGenerator = new BallGenerator(blueBallsGenVelocityFrom, blueBallsGenVelocityTo, blueBallsGenPos, 0.02, BLUE);
    lastTimeBallGenerated = -1;

    basket = new Basket(1.0f, 0.05f);
    totalBallsGeneratedCount = 0;

    inGame = true;
  }

  void generateBalls() {
    Ball* redBall = redBallGenerator->generateBall();
    Ball* blueBall = blueBallGenerator->generateBall();
    balls.push_back(redBall);
    balls.push_back(blueBall);
  }

  void move() {
    if (totalBallsGeneratedCount >= BALLS_MAX_COUNT && balls.size() == 0) {
      inGame = false;
      return;
    }
    std::cout << balls.size() << " Red count: " << redCount << " Blue count: " << blueCount << '\n';
    int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
    if ((lastTimeBallGenerated + TIME_BETWEEN_BALLS_GENERATION < timeSinceStart || lastTimeBallGenerated == -1) && totalBallsGeneratedCount < BALLS_MAX_COUNT) {
      generateBalls();
      totalBallsGeneratedCount += 2;
      lastTimeBallGenerated = timeSinceStart;
    }
    int deltaTime = 10; // timeSinceStart - oldTimeSinceStart;
    oldTimeSinceStart = timeSinceStart;
    std::list<Ball*> ballsToRemove;

    Ball::move(balls, deltaTime, platforms);

    //Ball::ResolveBallsCollisionWithPlatform(balls, *platform1);
    //Ball::ResolveBallsCollisionWithPlatform(balls, *platform2);
    //Ball::ResolveBallsCollisionWithPlatform(balls, *platform3);
    //Ball::ResolveBallsCollisions(balls);

    for (Ball* ball : balls) {
      bool touchBasket = ball->checkCollisionWithBasket(basket);
      if (touchBasket) {
        if (ball->getBallType() == RED) {
          redCount++;
        }
        else {
          blueCount++;
        }
      }
      if (ball->shouldBeDeleted() || touchBasket) {
        ballsToRemove.push_back(ball);
      }
    }

    for (Ball* ball : ballsToRemove) {
      balls.remove(ball);
    }


    if (aPressed) {
      platform1->rotateLeft();
      aPressed = false;
    }
    if (sPressed) {
      platform1->rotateRight();
      sPressed = false;
    }
    if (dPressed) {
      platform2->rotateLeft();
      dPressed = false;
    }
    if (fPressed) {
      platform2->rotateRight();
      fPressed = false;
    }
  }

  void draw() {
    if (!inGame) {
      drawResultScreen();
      return;
    }
    for (Ball* ball : balls) {
      ball->draw();
    }

    basket->draw();
    platform1->draw();
    platform2->draw();
    platform3->draw();

    if (balls.size() > 3) {
      /*const char *text = "Some sample text goes here.\n"
       "Yada yada yada, more text...\n"
       "foobar xyzzy\n";
      glColor3f(1, 0, 1);
      dtx_string(text);*/
     // font.setPenPosition(16, 16);
     // font.draw("Hello, gltext!");
    }
  }

  void handleKey(unsigned char c) {
    if (c == 'a') {
      aPressed = true;
    }
    if (c == 's') {
      sPressed = true;
    }
    if (c == 'd') {
      dPressed = true;
    }
    if (c == 'f') {
      fPressed = true;
    }
    if (c == 'i') {
      platform3->moveTo(0, 0.01f);
    }
    if (c == 'k') {
      platform3->moveTo(0, -0.01f);
    }
    if (c == 'j') {
      platform3->moveTo(-0.01f, 0);
    }
    if (c == 'l') {
      platform3->moveTo(0.01f, 0);
    }
  }

  void drawResultScreen() {
    if (wasResultShown) {
      return;
    }
    wasResultShown = true;
    if (redCount >= RED_BALLS_COUNT_TO_WIN && blueCount >= BLUE_BALLS_COUNT_TO_WIN) {
      std::cout << "You win!!\n";
    }
    else {
      std::cout << "You lose!!\n";
    }
  }

private:
  BallGenerator* redBallGenerator;
  BallGenerator* blueBallGenerator;
  std::list<Ball*> balls;
  std::list<Platform*> platforms;
  Basket* basket;
  Platform* platform1;
  Platform* platform2;
  Platform* platform3;
  int oldTimeSinceStart;
  int lastTimeBallGenerated;
  int redCount;
  int blueCount;
  int totalBallsGeneratedCount;
  bool inGame;
  bool wasResultShown = false;
  bool aPressed = false;
  bool sPressed = false;
  bool dPressed = false;
  bool fPressed = false;
  //gltext::Font font;
};