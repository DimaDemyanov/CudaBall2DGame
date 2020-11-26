#pragma once
#include <list>
#include <iostream>
#define GLT_IMPLEMENTATION
#include <gl/glew.h>
#include <gl/gl.h>
#include<gl/glut.h>
#include"Ball.h"
#include"BallGenerator.h"
#include"Basket.h"
#include"Platform.h"

const unsigned int TIME_BETWEEN_BALLS_GENERATION = 200;
const unsigned int BALLS_MAX_COUNT = 300;
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

    platform1 = new Platform(platform1Pos, 0.0f, 0.4f, 0.05f);
    platform2 = new Platform(platform2Pos, 0.0f, 0.45f, 0.07f);

    redBallGenerator = new BallGenerator(redBallsGenVelocityFrom, redBallsGenVelocityTo, redBallsGenPos, 0.03, RED);
    blueBallGenerator = new BallGenerator(blueBallsGenVelocityFrom, blueBallsGenVelocityTo, blueBallsGenPos, 0.04, BLUE);
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
    int deltaTime = timeSinceStart - oldTimeSinceStart;
    oldTimeSinceStart = timeSinceStart;
    std::list<Ball*> ballsToRemove;

    Ball::move(balls, deltaTime);

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
      if (ball->checkCollisionWithPlatform(platform1)) {
        ball->changeVelocity(platform1);
      }
      if (ball->checkCollisionWithPlatform(platform2)) {
        ball->changeVelocity(platform2);
      }
      if (ball->shouldBeDeleted() || touchBasket) {
        delete ball;
        ballsToRemove.push_back(ball);
      }
    }

    for (Ball* ball : ballsToRemove) {
        balls.remove(ball);
    }

    int i = 0;
    for (auto ballIt1 = balls.begin(); ballIt1 != balls.end(); ballIt1++, i++) {
      int j = 0;
      for (auto ballIt2 = balls.begin(); ballIt2 != balls.end(); ballIt2++, j++) {
        Ball* ball1 = *ballIt1;
        Ball* ball2 = *ballIt2;
        if (ball1->colliding(ball2) && i > j) {
          std::cout << "Colliding\n";
          ball1->resolveCollision(ball2);
        }
      }
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
      platform1->rotateLeft();
    }
    if (c == 's') {
      platform1->rotateRight();
    }
    if (c == 'd') {
      platform2->rotateLeft();
    }
    if (c == 'f') {
      platform2->rotateRight();
    }
  }

  void drawResultScreen() {
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
  Basket* basket;
  Platform* platform1;
  Platform* platform2;
  int oldTimeSinceStart;
  int lastTimeBallGenerated;
  int redCount;
  int blueCount;
  int totalBallsGeneratedCount;
  bool inGame;
  //gltext::Font font;
};