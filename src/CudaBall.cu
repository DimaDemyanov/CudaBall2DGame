#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"Ball.h"
#include"math.h"
#include<iostream>
#define gravity 0.2f

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

__global__ void CudaMoveBallsKernel(BallData* ballsData, int size, float deltaTime)
{
  int tId = blockIdx.x * blockDim.x + threadIdx.x;
  if (tId < size) {
    ballsData[tId].pos_x = ballsData[tId].pos_x - ballsData[tId].velocity_x * deltaTime;
    ballsData[tId].pos_y = ballsData[tId].pos_y - ballsData[tId].velocity_y * deltaTime;
    ballsData[tId].velocity_y += gravity * deltaTime;
  }
}

cudaError_t CudaMoveBallsKernelInvoke(BallData* ballsData, int size, float deltaTime)
{
  int threadsPerBlock = 1024;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  CudaMoveBallsKernel<<<blocksPerGrid, threadsPerBlock>>>(ballsData, size, deltaTime);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cout << "CudaMoveBallsKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}

// Resolving collisions between balls
__device__ void CudaResolveBallsCollision(BallData* ball1, BallData* ball2) {
  // get the mtd
  //Vector* delta = pos->subtract(ball->getPos());
  float delta_x = ball1->pos_x - ball2->pos_x;
  float delta_y = ball1->pos_y - ball2->pos_y;

  //float d = delta->getLength();
  float d = sqrt(delta_x * delta_x + delta_y * delta_y);
  
  // minimum translation distance to push balls apart after intersecting
  //Vector* mtd = delta->multiply(((getRadius() + ball->getRadius()) - d) / d);
  float mtd_x = delta_x * (((ball1->radius + ball2->radius) - d) / d);
  float mtd_y = delta_y * (((ball1->radius + ball2->radius) - d) / d);
  
  // resolve intersection --
  // inverse mass quantities
  //float im1 = 1 / getMass();
  //float im2 = 1 / ball->getMass();
  float im1 = 1 / ball1->mass;
  float im2 = 1 / ball2->mass;

  // push-pull them apart based off their mass
  //pos = pos->add(mtd->multiply(im1 / (im1 + im2)));
  //ball->pos = ball->getPos()->subtract(mtd->multiply(im2 / (im1 + im2)));

  // impact speed
  //Vector* v = getVelocity()->subtract(ball->getVelocity());
  float v_x = ball1->velocity_x - ball2->velocity_x;
  float v_y = ball1->velocity_y - ball2->velocity_y;

  //float vn = v->dot(mtd->normalize());
  float mtd_len = sqrt(mtd_x * mtd_x + mtd_y * mtd_y);
  float vn = v_x * mtd_x / mtd_len + v_y * mtd_y / mtd_len;

  // sphere intersecting but moving away from each other already
  if (vn < 0.0f) {
    return;
  }

  // collision impulse
  float i = (-(1.0f + 1.0f) * vn) / (im1 + im2);
  //Vector* impulse = mtd->normalize()->multiply(i);
  float impulse_x = mtd_x / mtd_len * i;
  float impulse_y = mtd_y / mtd_len * i;

  float elastic_coef = 0.5f;
  if (ball1->type == ball2->type) {
    elastic_coef = 0.5f;
  }

  // change in momentum
  //setVelocity(getVelocity()->add(impulse->multiply(im1)));
  ball1->velocity_x = ball1->velocity_x + impulse_x * im1 * elastic_coef;
  ball1->velocity_y = ball1->velocity_y + impulse_y * im1 * elastic_coef;

  //ball->setVelocity(ball->getVelocity()->subtract(impulse->multiply(im2)));
  ball2->velocity_x = ball2->velocity_x - impulse_x * im2 * elastic_coef;
  ball2->velocity_y = ball2->velocity_y - impulse_y * im2 * elastic_coef;
}

__global__ void CudaResolveBallsCollisionsKernel(BallData* ballsData, int size)
{
  int tId = blockIdx.x * blockDim.x + threadIdx.x;
  if (tId < size) {
    for (int i = tId + 1; i < size; i++) {
      float dist = sqrt((ballsData[tId].pos_x - ballsData[i].pos_x) * (ballsData[tId].pos_x - ballsData[i].pos_x) + (ballsData[tId].pos_y - ballsData[i].pos_y) * (ballsData[tId].pos_y - ballsData[i].pos_y));
      if (ballsData[tId].radius + ballsData[i].radius > dist) {
        CudaResolveBallsCollision(ballsData + tId, ballsData + i);
      }
    }
  }
}

cudaError_t CudaResolveBallsCollisionsKernelInvoke(BallData* ballsData, int size)
{
  int threadsPerBlock = 1024;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  CudaResolveBallsCollisionsKernel <<<blocksPerGrid, threadsPerBlock>>> (ballsData, size);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cout << "CudaResolveBallsCollisionsKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}

// Resolving collisions with Platform

__device__ void changeVelocity(float cx, float cy, float radius, float left, float top, float right, float bottom, float velocity_x, float velocity_y, float* newVelocity_x, float* newVelocity_y) {
  float closestX = (cx < left ? left : (cx > right ? right : cx));
  float closestY = (cy < top ? top : (cy > bottom ? bottom : cy));
  float dx = closestX - cx;
  float dy = closestY - cy;

  *newVelocity_x = velocity_x;
  *newVelocity_y = velocity_y;
  if (dy > dx && dx != 0) {
    if (dx * velocity_x > 0) {
      return;
    }
    *newVelocity_x = -velocity_x;
  }
  else {
    if (dy * velocity_y > 0) {
      return;
    }
    *newVelocity_y = -velocity_y;
  }
}

__device__ void changeVelocity(float cx, float cy, float radius, float x, float y, float width, float height, float angle, float velocity_x, float velocity_y, float* newVelocity_x, float* newVelocity_y) {
  float alpha = -angle / 180 * PI;
  float x1 = (cx - x) * cos(alpha) - (cy - y) * sin(alpha);
  float y1 = (cx - x) * sin(alpha) + (cy - y) * cos(alpha);
  //Vector* newVelocity = new Vector(velocity->getX() * cos(alpha) - velocity->getY() * sin(alpha), velocity->getX() * sin(alpha) + velocity->getY() * cos(alpha));
  *newVelocity_x = velocity_x * cos(alpha) - velocity_y * sin(alpha);
  *newVelocity_y = velocity_x * sin(alpha) + velocity_y * cos(alpha);
  changeVelocity(x1, y1, radius, -width / 2, height / 2, width / 2, -height / 2, *newVelocity_x, *newVelocity_y, newVelocity_x, newVelocity_y);
  float resultVelocity_x = *newVelocity_x * cos(alpha) + *newVelocity_y * sin(alpha);
  float resultVelocity_y = -*newVelocity_x * sin(alpha) + *newVelocity_y * cos(alpha);
  *newVelocity_x = resultVelocity_x;
  *newVelocity_y = resultVelocity_y;
}

__device__ bool intersectsBallAndRect(float cx, float cy, float radius, float left, float top, float right, float bottom) {
  float closestX = (cx < left ? left : (cx > right ? right : cx));
  float closestY = (cy > top ? top : (cy < bottom ? bottom : cy));
  float dx = closestX - cx;
  float dy = closestY - cy;

  return (dx * dx + dy * dy) <= radius * radius;
}

__device__ bool intersectsBallAndRotatedRect(float cx, float cy, float radius, float x, float y, float width, float height, float angle) {
  float alpha = -angle / 180 * PI;
  float x1 = (cx - x) * cos(alpha) - (cy - y) * sin(alpha);
  float y1 = (cx - x) * sin(alpha) + (cy - y) * cos(alpha);
  return intersectsBallAndRect(x1, y1, radius, -width / 2, height / 2, width / 2, -height / 2);
}

__global__ void CudaResolveBallsCollisionWithPlatformKernel(BallData* ballsData, int size, PlatformData platformData)
{
  int tId = blockIdx.x * blockDim.x + threadIdx.x;
  if (tId < size) {
    if (intersectsBallAndRotatedRect(ballsData[tId].pos_x, ballsData[tId].pos_y, ballsData[tId].radius, platformData.pos_x, platformData.pos_y, platformData.width, platformData.height, platformData.angle)) {
      changeVelocity(ballsData[tId].pos_x, ballsData[tId].pos_y, ballsData[tId].radius, platformData.pos_x, platformData.pos_y, platformData.width, platformData.height, platformData.angle, ballsData[tId].velocity_x, ballsData[tId].velocity_y, &(ballsData[tId].velocity_x), &(ballsData[tId].velocity_y));
    }
  }
}

cudaError_t CudaResolveBallsCollisionWithPlatformKernelInvoke(BallData* ballsData, int size, PlatformData platformData)
{
  int threadsPerBlock = 1024;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  CudaResolveBallsCollisionWithPlatformKernel << <blocksPerGrid, threadsPerBlock >> > (ballsData, size, platformData);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    std::cout << "CudaResolveBallsCollisionsKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
  }

  return cudaStatus;
}


void Ball::CudaMoveBalls(BallData* ballsData, int ballsCount, float deltaTime, PlatformData* platformData, int platformDataCount) {
  BallData* ballsDataGPU;

  allocateOnGPU((void**)&ballsDataGPU, ballsCount * sizeof(BallData));
  moveToGPU(ballsData, ballsDataGPU, ballsCount * sizeof(BallData));

  CudaMoveBallsKernelInvoke(ballsDataGPU, ballsCount, deltaTime);
  CudaResolveBallsCollisionsKernelInvoke(ballsDataGPU, ballsCount);
  for (int i = 0; i < platformDataCount; i++) {
    CudaResolveBallsCollisionWithPlatformKernelInvoke(ballsDataGPU, ballsCount, platformData[i]);
  }
  moveToCPU(ballsData, ballsDataGPU, ballsCount * sizeof(BallData));

  cudaFree(ballsDataGPU);
}

//void Ball::CudaResolveBallsCollisions(BallData* ballsData, int size) {
//  BallData* ballsDataGPU;
//
//  allocateOnGPU((void**)&ballsDataGPU, size * sizeof(BallData));
//  moveToGPU(ballsData, ballsDataGPU, size * sizeof(BallData));
//
//  CudaResolveBallsCollisionsKernelInvoke(ballsDataGPU, size);
//
//  moveToCPU(ballsData, ballsDataGPU, size * sizeof(BallData));
//  cudaFree(ballsDataGPU);
//}
//
//void Ball::CudaResolveBallsCollisionWithPlatform(BallData* ballsData, int size, PlatformData platformData) {
//  BallData* ballsDataGPU;
//
//  allocateOnGPU((void**)&ballsDataGPU, size * sizeof(BallData));
//  moveToGPU(ballsData, ballsDataGPU, size * sizeof(BallData));
//
//  CudaResolveBallsCollisionWithPlatformKernelInvoke(ballsDataGPU, size, platformData);
//
//  moveToCPU(ballsData, ballsDataGPU, size * sizeof(BallData));
//  cudaFree(ballsDataGPU);
//}