#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"Ball.h"
#include<iostream>

cudaError_t allocateOnGPU(void** arrayData, int size_in_bytes)
{
  cudaError_t cudaStatus = cudaMalloc((void**)arrayData, size_in_bytes);

  if (cudaStatus != cudaSuccess)
  {
    std::cout << "CudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}

cudaError_t moveToGPU(void* dataCPU, void* dataGPU, int size_in_bytes)
{
  cudaError_t cudaStatus = cudaMemcpy(dataGPU, dataCPU, size_in_bytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
  {
    std::cout << "TransferData from CPU to GPU failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}

cudaError_t moveToCPU(void* dataCPU, void* dataGPU, int size_in_bytes)
{
  cudaError_t cudaStatus = cudaMemcpy(dataCPU, dataGPU, size_in_bytes, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
  {
    std::cout << "TransferData from GPU to CPU failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}

__global__ void CudaMoveBallsKernel(BallData* ballsData, int size, int deltaTime)
{
  int tId = blockIdx.x*blockDim.x + threadIdx.x;
  ballsData[tId].pos_x = ballsData[tId].pos_x - ballsData[tId].velocity_x * deltaTime;
  ballsData[tId].pos_y = ballsData[tId].pos_y - ballsData[tId].velocity_y * deltaTime;
  //velocity->setY(velocity->getY() + g * deltatime);
}

cudaError_t CudaMoveBallsKernelInvoke(BallData* ballsData, int size, int deltaTime)
{
  int threadsPerBlock = 1024;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  CudaMoveBallsKernel<<<blocksPerGrid, threadsPerBlock>>>(ballsData, size, deltaTime);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cout << "sumKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}


void Ball::CudaMoveBalls(BallData* ballsData, int size, int deltaTime) {
  BallData* ballsDataGPU;

  allocateOnGPU((void**)&ballsDataGPU, size * sizeof(BallData));
  moveToGPU(ballsData, ballsDataGPU, size * sizeof(BallData));

  CudaMoveBallsKernelInvoke(ballsDataGPU, size, deltaTime);

  moveToCPU(ballsData, ballsDataGPU, size * sizeof(BallData));
}

bool Ball::CudaColliding(Ball* ball1, Ball * ball2)
{
  Vector* pos = ball1->getPos();
  Ball* ball = ball2;
  float xd = pos->getX() - ball->getPos()->getX();
  float yd = pos->getY() - ball->getPos()->getY();

  float sumRadius = ball1->getRadius() + ball->getRadius();
  float sqrRadius = sumRadius * sumRadius;

  float distSqr = (xd * xd) + (yd * yd);

  if (distSqr <= sqrRadius)
  {
    return true;
  }

  return false;
}

void Ball::CudaResolveCollision(Ball* ball1, Ball * ball2)
{
  Vector* pos = ball1->getPos();
  Ball* ball = ball2;
  // get the mtd
  Vector* delta = pos->subtract(ball->getPos());
  float d = delta->getLength();
  // minimum translation distance to push balls apart after intersecting
  Vector* mtd = delta->multiply(((ball1->getRadius() + ball->getRadius()) - d) / d);


  // resolve intersection --
  // inverse mass quantities
  float im1 = 1 / ball1->getMass();
  float im2 = 1 / ball->getMass();

  // push-pull them apart based off their mass
  //pos = pos->add(mtd->multiply(im1 / (im1 + im2)));
  //ball->pos = ball->getPos()->subtract(mtd->multiply(im2 / (im1 + im2)));

  // impact speed
  Vector* v = ball1->getVelocity()->subtract(ball->getVelocity());
  float vn = v->dot(mtd->normalize());

  // sphere intersecting but moving away from each other already
  if (vn < 0.0f) {
    return;
  }

  // collision impulse
  float i = (-(1.0f + 1.0f) * vn) / (im1 + im2);
  Vector* impulse = mtd->normalize()->multiply(i);

  // change in momentum
  ball1->setVelocity(ball1->getVelocity()->add(impulse->multiply(im1)));
  ball->setVelocity(ball->getVelocity()->subtract(impulse->multiply(im2)));

}