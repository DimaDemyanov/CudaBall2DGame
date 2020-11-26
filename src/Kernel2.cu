#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <cstdio>

#define EPS 0.00000001

const float RANDOM_FROM = 0;
const float RANDOM_TO = 10;

__global__ void sumKernel(const float *a, const float *b, float* result, int num)
{

  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  float tmpSum = 0;

  if (row < num && col < num) {
    for (int i = 0; i < num; i++) {
      tmpSum += a[row * num + i] * b[i * num + col];
    }
  }
  result[row * num + col] = tmpSum;
}

cudaError_t allocateOnGPU(float** arrayData, int num)
{
	size_t size = num * sizeof(float);
	cudaError_t cudaStatus = cudaMalloc((void**)arrayData, size);

	if (cudaStatus != cudaSuccess)
	{
		std::cout << "CudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	}

	return cudaStatus;
}

int allocateOnCPU(float** arrayData, int num)
{
	*arrayData = new float[num];

	if (arrayData == NULL)
	{
		std::cout << "Malloc failed!" << std::endl;
		return 1;
	}
	return 0;
}

void createRandomData(float* data, int num, float start, float finish)
{
	for (int i = 0; i < num; i++){
		data[i] = (finish - start) * (float)(rand() / (float)RAND_MAX) + start;
	}
}

void createZerosData(float* data, int num)
{
	for (int i = 0; i < num; i++)
	{
		data[i] = 0.0f;
	}
}

cudaError_t moveToGPU(float* dataCPU, float* dataGPU, int num)
{
	cudaError_t cudaStatus = cudaMemcpy(dataGPU, dataCPU, num * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "TransferData from CPU to GPU failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	}

	return cudaStatus;
}

cudaError_t moveToCPU(float* dataCPU, float* dataGPU, int num)
{
	cudaError_t cudaStatus = cudaMemcpy(dataCPU, dataGPU, num * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "TransferData from GPU to CPU failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	}

	return cudaStatus;
}

cudaError_t kernel(float* a, float* b, float* result, int num)
{
	int threadsPerBlock = 1024;
	int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
	
	//sumKernel << <blocksPerGrid, threadsPerBlock >> > (a, b, result, num);
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "sumKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	}

	return cudaStatus;
}

int chechResult(float* res1, float* res2, int num)
{
	for (int i = 0; i < num * num; i++)
	{
    if (fabs(res1[i] - res2[i]) > EPS) {
      return 1;
    }
	}
	return 0;
}

float * multiplyMatrixesOnCPU(float* a, float* b, int num) {
  float* result = new float[num * num];

  for (int i = 0; i < num; i++)
  {
    for (int j = 0; j < num; j++)
    {
      result[i * num + j] = 0;
      for (int k = 0; k < num; k++)
      {
        result[i * num + j] += a[i * num + k] * b[k * num + j];
      }
    }
  }
  return result;
}

void printMatrix(float* a, int num) {
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < num; j++) {
      std::cout << a[i * num + j] << " ";
    }
    std::cout << '\n';
  }
}

int main()
{
	std::srand(unsigned(time(NULL)));  

  int num = 5;

	float *aCPU = NULL;
	float *bCPU = NULL;
	float *cCPU = NULL;
		
	float *aGPU = NULL;
	float *bGPU = NULL;
	float *cGPU = NULL;
		
	if (allocateOnCPU(&aCPU, num) == 1)
		return 0;

	if (allocateOnCPU(&bCPU, num) == 1)
		return 0;

	if (allocateOnCPU(&cCPU, num) == 1)
		return 0;

	createRandomData(aCPU, num, RANDOM_FROM, RANDOM_TO);
	createRandomData(bCPU, num, RANDOM_FROM, RANDOM_TO);
	createZerosData(cCPU, num);

	allocateOnGPU(&aGPU, num);
	allocateOnGPU(&bGPU, num);
	allocateOnGPU(&cGPU, num);

	moveToGPU(aCPU, aGPU, num);
  moveToGPU(bCPU, bGPU, num);
  moveToGPU(cCPU, cGPU, num);

	kernel(aGPU, bGPU, cGPU, num);

	moveToCPU(cCPU, cGPU, num);

  float* CPUresult = multiplyMatrixesOnCPU(aCPU, bCPU, num);

  if (chechResult(CPUresult, cCPU, num) == 1) {
    std::cout << "Sum is not the same for CPU and GPU" << std::endl;
  } else {
    std::cout << "Sum is the same for CPU and GPU" << std::endl;
  }

  std::cout << "A:";
  printMatrix(aCPU, num);

  std::cout << "B:";
  printMatrix(bCPU, num);

  std::cout << "Result on CPU:";
  printMatrix(CPUresult, num);

  std::cout << "Result on GPU:";
  printMatrix(cCPU, num);

	delete[] aCPU;
	delete[] bCPU;
	delete[] cCPU;

	cudaFree(aGPU);
	cudaFree(bGPU);
	cudaFree(cGPU);

  std::getchar();

	return 0;
}