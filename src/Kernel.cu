#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <device_functions.h>

#include "curand_kernel.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#include<glm/glm.hpp>
namespace spec_struct {
  struct Vec2
  {
    float x, y;

    __host__ __device__ Vec2 operator-(Vec2 T) {
      return Vec2(x - T.x, y - T.y);
    }
    __host__ __device__ Vec2 operator+(Vec2 T) {
      return Vec2(x + T.x, y + T.y);
    }
    __host__ __device__ float operator*(Vec2 T) {
      return x * T.x + y * T.y;
    }
    __host__ __device__ Vec2 operator*(float T) {
      return Vec2(x * T, y * T);
    }
    __host__ __device__ Vec2 operator=(glm::vec2 T) {
      x = T.x;
      y = T.y;
      return *this;
    }
    __host__ __device__ Vec2 operator+=(Vec2 T) {
      x += T.x;
      y += T.y;
      return *this;
    }
    __host__ __device__ bool operator==(Vec2 T) {
      return ((x == T.x) && (y == T.y));
    }

    __host__ __device__ glm::vec2 to_glm() {
      return glm::vec2(x, y);
    }

    __host__ __device__ float project_on_me(Vec2 source) {
      float magn = sqrtf(x * x + y * y);
      float sinA = y / magn;
      float cosA = x / magn;

      Vec2 prog;
      prog.x = source.x * cosA + source.y * sinA;
      prog.y = source.y * cosA - source.x * sinA;

      return sqrtf(prog.x * prog.x + prog.y * prog.y);
    }

    __host__ __device__ float mag() {
      return sqrtf((x * x + y * y));
    }

    __host__ __device__ Vec2 normalized() {
      float inv_mag = 1 / (this->mag());
      Vec2 vec = *this;

      vec.x *= inv_mag;
      vec.y *= inv_mag;

      return vec;
    }

    __host__ __device__ Vec2() {
      x = 0.0f;
      y = 0.0f;
    }
    __host__ __device__ Vec2(float _x) {
      x = _x;
      y = _x;
    }
    __host__ __device__ Vec2(glm::vec2 vec) {
      x = vec.x;
      y = vec.y;
    }
    __host__ __device__ Vec2(float _x, float _y) {
      x = _x;
      y = _y;
    }

  };
}

#define W_SHAPE_MOVE_PROC    0x01; // шестнадцатеричный литерал для 0000 0001
#define W_SHAPE_ROTATE_PROC  0x02; // шестнадцатеричный литерал для 0000 0010

typedef spec_struct::Vec2 Vec2;
#include "OpenGL.h"
#include "cuda_gl_interop.h"

#define PART_GEN_DEBUG

//#include <Windows.h>
//#include <string>

typedef unsigned int uint;

#pragma region Отладка


namespace Debug {

#ifdef PART_GEN_DEBUG
  bool isDebug = false;
#else
  bool isDebug = false;
#endif // PART_GEN_DEBUG == 1


  void Message(LPCWSTR text) {
    if (isDebug)
      if (MessageBoxW(NULL, text, L"Отладка CUDA", MB_OKCANCEL) == IDCANCEL)
        isDebug = false;
  }

  void Message(const char* text) {
    if (isDebug)
      if (MessageBox(NULL, text, "Отладка CUDA", MB_OKCANCEL) == IDCANCEL)
        isDebug = false;
  }
}

#pragma endregion

struct ball_data_s; // 32-bit
struct grid_data //16-bit
{
  glm::ivec2 grid_dimentions;
  glm::ivec2 cell_dimentions;
}
myGridData;

struct shape
{
  glm::vec2   point_a;
  glm::vec2   point_b;
  glm::vec2   point_c;
  glm::vec2   point_d;
  float       radius;
  bool        isRectangle;
  float       angle;
  char        flag;
};

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

# define M_PI           3.14159265358979323846  /* pi */
#include "math.h"
#include <list>
#include <sstream>
#include <chrono>
#include <iostream>

#include "Structures.h"
#include "Functions.h"

#include "ParticleSystem.h"

#include "ParticleGenerator_GL.h"


void RunCudaTest(Vec2 mouse_shift,
  Vec2 mouse_coord,
  int mouse_key,
  dev::strct::EnviromentData* common,
  int size);

void ComputeCollision(dev::strct::EnviromentData* env_data,
  ball_data_s* balls,
  float* interval,
  int size);

#include "Main.h"
__device__ int getNextIndexForCollision(int start_index, int curr_index, int current_cell_num, int num_of_particles, ball_data_s* particles_array)
{
  if (curr_index >= start_index)
  {
    if (curr_index != num_of_particles && particles_array[curr_index + 1].attachedCellIdx == current_cell_num)
    {
      return curr_index + 1;
    }
    else
    {
      if (start_index != 0 && particles_array[start_index - 1].attachedCellIdx == current_cell_num)
      {
        return start_index - 1;
      }
    }
  }
  else
  {
    if (curr_index != 0 && particles_array[curr_index - 1].attachedCellIdx == current_cell_num)
    {
      return curr_index - 1;
    }
  }

  return -1;
}

__device__ bool isPointInTriangle(Vec2 p, glm::vec2 t_1, glm::vec2 t_2, glm::vec2 t_3)
{
  float k, m, n;
  k = (t_1.x - p.x) * (t_2.y - t_1.y) - (t_2.x - t_1.x) * (t_1.y - p.y);
  m = (t_2.x - p.x) * (t_3.y - t_2.y) - (t_3.x - t_2.x) * (t_2.y - p.y);
  n = (t_3.x - p.x) * (t_1.y - t_3.y) - (t_1.x - t_3.x) * (t_3.y - p.y);

  return (k >= 0 && m >= 0 && n >= 0) || (k <= 0 && m <= 0 && n <= 0);
}

__device__ bool isPointInSphere(Vec2 p, glm::vec2 center, float radius)
{
  Vec2 c(center.x, center.y);

  p.x -= c.x;
  p.y -= c.y;

  return sqrtf(powf(p.x, 2) + powf(p.y, 2)) < radius;
}

__device__ float getImpulse(float k, float x) {
  const float h = k * x;
  return h * expf(1.0f - h);
}

__device__ shape* getShapeFromPoint(Vec2 point, shape* begin, shape* end)
{
  shape* ptr = begin;
  bool run_result;

  while (ptr != end)
  {
    run_result = false;
    if (ptr->isRectangle)
    {
      run_result = isPointInTriangle(point, ptr->point_a, ptr->point_b, ptr->point_c);
      run_result = run_result || isPointInTriangle(point, ptr->point_a, ptr->point_d, ptr->point_c);
    }
    else
    {
      run_result = isPointInSphere;
    }

    return (run_result ? ptr : nullptr);
    ptr++;
  }

  return nullptr;
}

__device__ Vec2 rand_vec2(Rect param, int idx)
{
  Vec2 result;
  float magnitude, angle;

  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> dist_magn(param.width, param.height);
  thrust::uniform_real_distribution<float> dist_angle(param.x, param.y);
  rng.discard(idx);

  magnitude = dist_magn(rng);
  angle = dist_angle(rng);

  result.x = magnitude * cosf(angle);
  result.y = magnitude * sinf(angle);

  return result;
}

__forceinline__ __device__ Vec2 pointRotate(float angle_rad, Vec2 center, Vec2 point) {

  float a_x = point.x;
  float a_y = point.y;

  float x = center.x;
  float y = center.y;

  point.x = x + (a_x - x) * cosf(angle_rad) - (a_y - y) * sinf(angle_rad);
  point.y = y + (a_x - x) * sinf(angle_rad) + (a_y - y) * cosf(angle_rad);

  return point;
}

__forceinline__ __device__ Vec2 getCenterPoint(Vec2 p1, Vec2 p2) {
  return Vec2((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

__forceinline__ __device__ Vec2 mtpByNormal(Vec2 base, Vec2 projected, float multiplyer) {

  float angle = acosf(base.x / base.mag());

  pointRotate(-angle, Vec2(0, 0), projected);

  return pointRotate(angle, Vec2(0, 0), ((projected.x > 0) ? Vec2(projected.x, projected.y) : Vec2(-projected.x, projected.y)));
}

__forceinline__ __device__ bool collideRectangle(const shape shape_s, const Vec2 point_s, float radius,
  Vec2* normal, float* delta)
{
  if (shape_s.isRectangle == true)
  {
    char collision_flags;

    Vec2 center = getCenterPoint(shape_s.point_a, shape_s.point_c);

    Vec2 p1 = (pointRotate(-shape_s.angle, center, shape_s.point_a) - center).to_glm();
    Vec2 p2 = (pointRotate(-shape_s.angle, center, shape_s.point_c) - center).to_glm();

    Vec2 a_p1 = (pointRotate(-shape_s.angle, center, shape_s.point_b) - center).to_glm();
    Vec2 a_p2 = (pointRotate(-shape_s.angle, center, shape_s.point_d) - center).to_glm();

    Vec2 point = (pointRotate(-shape_s.angle, center, point_s) - center);

    collision_flags |= (((p1.x) < point.x) ? 0x01 : 0x00);
    collision_flags |= (((p1.y) < point.y) ? 0x02 : 0x00);
    collision_flags |= (((p2.x) < point.x) ? 0x04 : 0x00);
    collision_flags |= (((p2.y) < point.y) ? 0x08 : 0x00);

    //*normal = Vec2(0, 0);

    switch (collision_flags)
    {
    case(0x0B):
      *delta = ((point.y - p1.y) - radius);
      if (*delta < 0)
        *normal = Vec2(0, 1);
      break;
    case(0x0F):
      *delta = (abs((point - a_p1).mag()) - radius);
      if (*delta < 0)
        *normal = Vec2(0.71, 0.71);
      break;
    case(0x0D):
      *delta = ((point.x - p2.x) - radius);
      if (*delta < 0)
        *normal = Vec2(1, 0);
      break;
    case(0x05):
      *delta = (abs((point - p2).mag()) - radius);
      if (*delta < 0)
        *normal = Vec2(0.71, -0.71);
      break;
    case(0x01):
      *delta = ((p2.y - point.y) - radius);
      if (*delta < 0)
        *normal = Vec2(0, -0.71);
      break;
    case(0x00):
      *delta = (abs((point - a_p2).mag()) - radius);
      if (*delta < 0)
        *normal = Vec2(-0.71, -0.71);
      break;
    case(0x08):
      *delta = ((p1.x - point.x) - radius);
      if (*delta < 0)
        *normal = Vec2(-1, 0);
      break;
    case(0x0A):
      *delta = (abs((point - p1).mag()) - radius);
      if (*delta < 0)
        *normal = Vec2(-0.71, 0.71);
      break;
    case(0x09):
      *normal = point.normalized();
      break;
    default:
      *normal = Vec2(0, 0);
      break;
    }
    if (*normal == Vec2(0, 0)) return false;

    *normal = pointRotate(shape_s.angle, Vec2(0, 0), *normal);
    //*normal = normal->normalized();
    return true;
  }
  else
  {
    Vec2 center = shape_s.point_a;
    Vec2 tmp_normal;

    float dist = sqrtf(powf((point_s.x - center.x), 2) + powf((point_s.y - center.y), 2));

    if (dist >= radius + shape_s.radius)
      return false;

    tmp_normal.x = (point_s.x - center.x) / dist;
    tmp_normal.y = (point_s.y - center.y) / dist;

    *normal = tmp_normal;

    return true;
  }
}

//__forceinline__ __device__ bool collideCircle(const shape circle_s, const Vec2 point_s, float radius,
//    Vec2* normal, float* delta)
//{
//    Vec2 center = circle_s.point_a;
//    Vec2 tmp_normal;
//
//    float dist = sqrtf(powf((point_s.x - center.x), 2) + powf((point_s.y - center.y), 2));
//
//    if (dist >= radius + circle_s.radius)
//        return false;
//
//    tmp_normal.x = (point_s.x - center.x) / dist;
//    tmp_normal.y = (point_s.y - center.y) / dist;
//
//    * normal = tmp_normal;
//
//    return true;
//}

__forceinline__ __device__ bool getNormalToLine(Vec2 p1, Vec2 p2,
  Vec2 point, Vec2* normal,
  float* delta, float radius,
  Vec2 center)
{
  double x1 = p1.x, x2 = p2.x, x3 = point.x, x4, x5;
  double y1 = p1.y, y2 = p2.y, y3 = point.y, y4, y5;

  x4 = ((x2 - x1) * (y2 - y1) * (y3 - y1) + x1 * powf(y2 - y1, 2) + x3 * powf(x2 - x1, 2)) / (powf(y2 - y1, 2) + powf(x2 - x1, 2));
  y4 = (y2 - y1) * (x4 - x1) / (x2 - x1) + y1;

  double a = 1;//expf((point - Vec2(x4, y4)).mag());

  *delta = (point - Vec2(x4, y4)).mag();

  x5 = a * (x4 - x3) / (y4 - y3) + x4;
  y5 = (y4 - y3) * (x5 - x4) / (x4 - x3) + y4;

  double s1 = (x4 - p2.x) / (p1.x - p2.x);

  *normal = Vec2(x5 - x4, y5 - y4);

  *normal = *normal * normal->project_on_me(point - center);

  *normal = normal->normalized();

  if (*delta < radius && ((s1 * p1.y + (1 - s1) * p2.y == y4) && (s1 >= 0) && (s1 <= 1)))
    return true;
  return false;
}

__forceinline__ __device__ bool lineCollided(Vec2 p1, Vec2 p2, Vec2 point, float radius) {

  float a, b;
  float x0 = point.x, y0 = point.y, r0 = radius;

  {
    float x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;

    a = (y1 - y2) / (x1 - x2);
    b = y2 - a * x2;
  }

  float k, l, m;
  k = powf(a, 2.f) + 1;
  l = 2 * a * (b - y0) - 2 * x0;
  m = powf(x0, 2.f) - powf(r0, 2.f) + powf(b - y0, 2.f);
  float D = powf(l, 2.f) - 4. * k * m;

  if (D >= 0) {
    Vec2 cros_p1; //1-ая точка пересечения
    Vec2 cros_p2; //2-ая точка пересечения

    cros_p1.x = (-l - sqrtf(D)) / (2 * k);
    cros_p1.y = a * cros_p1.x + b;
    cros_p2.x = (-l + sqrtf(D)) / (2 * k);
    cros_p2.y = a * cros_p2.x + b;

    float s1 = (cros_p1.x - p2.x) / (p1.x - p2.x);
    float s2 = (cros_p2.x - p2.x) / (p1.x - p2.x);

    if (((s1 * p1.y + (1 - s1) * p2.y == cros_p1.y) && (s1 >= 0) && (s1 <= 1)) || ((s2 * p1.y + (1 - s2) * p2.y == cros_p2.y) && (s2 >= 0) && (s2 <= 1)))
      return true;
  }

  return false;
}

__forceinline__ __device__ uint getCellNum(glm::vec2 cell_coord, grid_data grid) {
  if (
    (cell_coord.x >= 0) &&
    (cell_coord.x < grid.grid_dimentions.x) &&
    (cell_coord.y >= 0) &&
    (cell_coord.y < grid.grid_dimentions.y)
    )
    return cell_coord.x + cell_coord.y * grid.grid_dimentions.x;
  return -1;
}

__forceinline__ __device__ void getCells(glm::vec2 coord, grid_data grid, int* cells) {
  glm::ivec2 icell;

  icell.x = (int(coord.x) / (grid.cell_dimentions.x));
  icell.y = (int(coord.y) / (grid.cell_dimentions.y));

  cells[0] = getCellNum(glm::ivec2(icell.x, icell.y), grid); //ZERO
  cells[7] = getCellNum(glm::ivec2(icell.x - 1, icell.y - 1), grid); //LEFT BOTTOM
  cells[1] = getCellNum(glm::ivec2(icell.x - 1, icell.y + 1), grid); //LEFT TOP
  cells[8] = getCellNum(glm::ivec2(icell.x - 1, icell.y), grid);		//LEFT MID
  cells[5] = getCellNum(glm::ivec2(icell.x + 1, icell.y - 1), grid); //RIGHT BOTTOM
  cells[3] = getCellNum(glm::ivec2(icell.x + 1, icell.y + 1), grid); //RIGHT TOP
  cells[4] = getCellNum(glm::ivec2(icell.x + 1, icell.y), grid);		//RIGHT MID
  cells[6] = getCellNum(glm::ivec2(icell.x, icell.y - 1), grid);		//BOT
  cells[2] = getCellNum(glm::ivec2(icell.x, icell.y + 1), grid);		//TOP
}

__forceinline__ __device__ int getCellParticleIndex(int key, int ball_size, ball_data_s* data) {
  bool flag = false;

  if (key < 0) return -1;

  int l = 0; // левая граница
  int r = int(ball_size) - 1; // правая граница
  int mid;
  //if((ball_size == 0))
  while ((l <= r) && (flag != true)) {
    mid = (l + r) / 2; // считываем срединный индекс отрезка [l,r]

    if (data[mid].attachedCellIdx == key) flag = true; //проверяем ключ со серединным элементом
    if (data[mid].attachedCellIdx > key) r = mid - 1; // проверяем, какую часть нужно отбросить
    else l = mid + 1;
  }
  if (flag) {
    return mid;
  }
  else {
    return -1;
  }
}

__forceinline__ __device__ float checkParticleCollision(Vec2 p1, Vec2 p2, float radius)
{
  float dist = sqrtf(powf((p2.x - p1.x), 2) + powf((p2.y - p1.y), 2));

  if (fabsf(dist - 2 * radius) <= 2) // если столкновение есть
  {
    return dist;
  }
  return -1;
}

__device__ Vec2 calculateParticleColission(Vec2 p1, Vec2 p2, Vec2 speed_p1, Vec2 speed_p2, float dist)
{
  float dist_x = fabsf(p2.x - p1.x); //длина проекции dist на ОХ
  float dist_y = fabsf(p2.y - p1.y); //длина проекции dist на ОY

  float cosA = dist_x / dist; //A - угол, на которую будем поворачивать локальную систему координат частиц
  float sinA = dist_y / dist;

  Vec2 temp_speed_p1; //скорости частиц в проекциях на повернутую систему координат
  Vec2 temp_speed_p2;

  temp_speed_p1.y = speed_p1.y * cosA - speed_p1.x * sinA; //рассчет скорости частицы p1 в проекциях на повернутую систему координат
  temp_speed_p2.x = speed_p2.x * cosA + speed_p2.y * sinA;
  temp_speed_p1.x = temp_speed_p2.x;

  return { temp_speed_p1.x * cosA - temp_speed_p1.y * sinA , temp_speed_p1.x * sinA + temp_speed_p1.y * cosA };
  //return speed_p1;

}

//Код ядра
__global__ void PhysicsKernel(dev::strct::ParticleData* __restrict__ particles,
  dev::strct::CommonData* __restrict__ common,
  dev::strct::EnviromentData* __restrict__ env_comm,
  ball_data_s* __restrict__ balls,
  float* interval,
  int GeneratorNumber,
  int size)
{
  int p_idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int b_idx = p_idx + common->offcet;

  if (p_idx < size) {

    if (particles[p_idx].isActive == false && particles[p_idx].delayTimeNow >= particles[p_idx].delayTime) //если задержка сейчас равна зарержке, то частица вылетает из источника
    {
      particles[p_idx].isActive = true;
      particles[p_idx].delayTimeNow = 0; //задержка сейчас равна нулю, так как частица вылетает, а не ждет
      particles[p_idx].lifeTimeNow = 0; //начинает идти время жизни
      particles[p_idx].speed = rand_vec2(common->generator_params, p_idx);
      particles[p_idx].coord = common->spawnPoint;

      balls[b_idx].attachedCellIdx = -1000;
      env_comm->keys_array[b_idx] = balls[b_idx].attachedCellIdx;

    }
    else
    {
      if (particles[p_idx].isActive == false /*&& particles[p_idx].delayTimeNow >= 0 && particles[p_idx].delayTimeNow < particles[p_idx].delayTime*/) //если частица ждет на спавне
      {
        particles[p_idx].delayTimeNow += *interval;

        balls[b_idx].attachedCellIdx = -1000;
        env_comm->keys_array[b_idx] = balls[b_idx].attachedCellIdx;
      }
    }

    if (particles[p_idx].isActive == true && particles[p_idx].lifeTimeNow >= common->lifeTime) //если время жизни вышло
    {
      particles[p_idx].isActive = false;
      particles[p_idx].coord = common->spawnPoint;
      particles[p_idx].lifeTimeNow = 0;
      particles[p_idx].delayTimeNow = 0;

      balls[b_idx].attachedCellIdx = -1000;
      env_comm->keys_array[b_idx] = balls[b_idx].attachedCellIdx;
    }
    else //если частица летает
    {
      if (particles[p_idx].isActive == true /*&& particles[p_idx].lifeTimeNow >= 0*/)
      {
        particles[p_idx].lifeTimeNow += *interval;
        particles[p_idx].speed.y += common->gravity;

        particles[p_idx].coord.x += particles[p_idx].speed.x * *interval;
        particles[p_idx].coord.y += particles[p_idx].speed.y * *interval;

        balls[b_idx].attachedCellIdx = (particles[p_idx].coord.x / env_comm->grid.cell_dimentions.x) + (int)(particles[p_idx].coord.y / env_comm->grid.cell_dimentions.y) * (env_comm->grid.grid_dimentions.x);
        env_comm->keys_array[b_idx] = balls[b_idx].attachedCellIdx;
      }

    }

    balls[b_idx].coord.x = particles[p_idx].coord.x;
    balls[b_idx].coord.y = particles[p_idx].coord.y;

    balls[b_idx].particle_idx = p_idx;
    balls[b_idx].generatorSource = GeneratorNumber;
  }
}

__global__ void CollisionKernel(dev::strct::EnviromentData*  __restrict__ env_data,
  ball_data_s* __restrict__ balls,
  float* interval,
  int size)
{
  int p_idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int gen_idx, cell_idx, b_idx;

  *(env_data->time_test) += *interval;

  dev::strct::ParticleData* particle;

  if (p_idx < size && balls[p_idx].generatorSource != -1)
  {
    gen_idx = balls[p_idx].generatorSource;
    cell_idx = balls[p_idx].attachedCellIdx;
    b_idx = p_idx;
    p_idx = balls[p_idx].particle_idx;

    particle = &(env_data->ParticleDataSet.pointer[gen_idx][p_idx]);
    if (cell_idx >= 0) {


      for (int i = 0; i < 5; i++) {

        Vec2 norm;
        float delta;

        if (collideRectangle(env_data->shapes[i],
          particle->coord,
          env_data->p_radius,
          &norm, &delta))
        {
          //norm = norm.normalized();
          particle->speed = particle->speed + norm * norm.project_on_me(particle->speed * 0.5);
          particle->speed = particle->speed * env_data->CommonDataSet.pointer[gen_idx]->k;
        }
      }
      __syncthreads();


      int cells[9];
      Vec2 temp_speed = 0;
      float temp_dist;
      bool colided = false;
      int counter = 0;

      getCells(particle->coord.to_glm(), env_data->grid, cells);

      int start_index;
      int curr_index;

      for (int i = 0; i < 9; i++) {
        start_index = getCellParticleIndex(int(cells[i]), size, balls);
        curr_index = start_index;

        if (curr_index == -1) //если i-ая ячейка оказалась пустой
        {
          continue;
        }

        while (curr_index != -1)
        {

          counter++;
          //temp_dist = 0;
          // DON'T FORGOT ABOUT FIRST PROCESS HERE !!!!
          if (balls[curr_index].generatorSource != gen_idx)
          {

            temp_dist = checkParticleCollision(particle->coord, balls[curr_index].coord, env_data->p_radius);
            if (temp_dist != -1)
            {
              temp_speed = temp_speed + calculateParticleColission(particle->coord, balls[curr_index].coord, particle->speed,
                env_data->ParticleDataSet.pointer[balls[curr_index].generatorSource][balls[curr_index].particle_idx].speed, temp_dist);
              colided = true;

            }
          }
          curr_index = getNextIndexForCollision(start_index, curr_index, int(cells[i]), size, balls);
        }
      }

      __syncthreads();

      if (colided == true) {
        particle->speed = temp_speed.normalized() * particle->speed.mag() * 0.98f;
        //particle->speed = temp_speed;

      }


      if (counter < 10) counter == 10;
      float kf = counter / 100.0;
      if (gen_idx == 0)
      {
        balls[b_idx].color = glm::vec4(0.031f, 0.0f, 1.0f, 1.0f);
      }
      else
      {
        balls[b_idx].color = glm::vec4(1.0f, 0.0f, 0.031f, 1.0f);
      }


      balls[b_idx].color = balls[b_idx].color * kf;
      balls[b_idx].color.a = 1.0;

    }
  }
}

__forceinline__ __device__ void rectRotate(shape* shape, float angle_rad, Vec2 point) {

  float a_x = shape->point_a.x;
  float b_x = shape->point_b.x;
  float c_x = shape->point_c.x;
  float d_x = shape->point_d.x;
  float a_y = shape->point_a.y;
  float b_y = shape->point_b.y;
  float c_y = shape->point_c.y;
  float d_y = shape->point_d.y;

  float x = point.x;
  float y = point.y;

  shape->point_a.x = x + (a_x - x) * cosf(angle_rad) - (a_y - y) * sinf(angle_rad);
  shape->point_b.x = x + (b_x - x) * cosf(angle_rad) - (b_y - y) * sinf(angle_rad);
  shape->point_c.x = x + (c_x - x) * cosf(angle_rad) - (c_y - y) * sinf(angle_rad);
  shape->point_d.x = x + (d_x - x) * cosf(angle_rad) - (d_y - y) * sinf(angle_rad);

  shape->point_a.y = y + (a_x - x) * sinf(angle_rad) + (a_y - y) * cosf(angle_rad);
  shape->point_b.y = y + (b_x - x) * sinf(angle_rad) + (b_y - y) * cosf(angle_rad);
  shape->point_c.y = y + (c_x - x) * sinf(angle_rad) + (c_y - y) * cosf(angle_rad);
  shape->point_d.y = y + (d_x - x) * sinf(angle_rad) + (d_y - y) * cosf(angle_rad);
}

__global__ void TestKernel(Vec2 mouse_shift, Vec2 mouse_coord, int mouse_key, dev::strct::EnviromentData* common, int size)
{
  int p_idx = (threadIdx.x + blockIdx.x * blockDim.x);

  if (p_idx < size) {

    if (!common->shapes_init) {
      //common->shapes[p_idx].flag = 0;

      if (p_idx == 0)
      {
        common->shapes[p_idx].isRectangle = true;
        common->shapes[p_idx].point_a = Vec2(50 + p_idx * 100, 0).to_glm();
        common->shapes[p_idx].point_b = Vec2(200 + p_idx * 100, 0).to_glm();
        common->shapes[p_idx].point_c = Vec2(200 + p_idx * 100, 50).to_glm();
        common->shapes[p_idx].point_d = Vec2(50 + p_idx * 100, 50).to_glm();
      }
      else
      {
        if (p_idx % 2 == 0)
        {
          common->shapes[p_idx].isRectangle = false;
          common->shapes[p_idx].point_a = Vec2(50 + p_idx * 100, 100 + p_idx * 100).to_glm();
          common->shapes[p_idx].radius = 20;
        }
        else
        {
          common->shapes[p_idx].isRectangle = true;
          common->shapes[p_idx].point_a = Vec2(50 + p_idx * 100, 100 + p_idx * 100).to_glm();
          common->shapes[p_idx].point_b = Vec2(200 + p_idx * 100, 100 + p_idx * 100).to_glm();
          common->shapes[p_idx].point_c = Vec2(200 + p_idx * 100, 50 + p_idx * 100).to_glm();
          common->shapes[p_idx].point_d = Vec2(50 + p_idx * 100, 50 + p_idx * 100).to_glm();
        }
      }

    }

    //shape* target = &(common->shapes[p_idx]);
    bool process;

    if (mouse_key == 1) {

      if (common->shapes[p_idx].isRectangle)
      {
        process = isPointInTriangle(mouse_coord, common->shapes[p_idx].point_a, common->shapes[p_idx].point_b, common->shapes[p_idx].point_c);
        process = process || isPointInTriangle(mouse_coord, common->shapes[p_idx].point_a, common->shapes[p_idx].point_d, common->shapes[p_idx].point_c);
      }
      else
      {
        process = isPointInSphere(mouse_coord, common->shapes[p_idx].point_a, common->shapes[p_idx].radius);
      }

      __syncthreads();

      if (process || (common->shapes[p_idx].flag & 0x01))
      {
        common->shapes[p_idx].flag |= W_SHAPE_MOVE_PROC;

        common->shapes[p_idx].point_a.x += mouse_shift.x; //mouse_shift.x
        common->shapes[p_idx].point_b.x += mouse_shift.x;
        common->shapes[p_idx].point_c.x += mouse_shift.x;
        common->shapes[p_idx].point_d.x += mouse_shift.x;

        common->shapes[p_idx].point_a.y += mouse_shift.y;
        common->shapes[p_idx].point_b.y += mouse_shift.y;
        common->shapes[p_idx].point_c.y += mouse_shift.y;
        common->shapes[p_idx].point_d.y += mouse_shift.y;
      }
    }
    else
    {
      common->shapes[p_idx].flag &= ~W_SHAPE_MOVE_PROC;
    }

    if (mouse_key == 2)
    {
      if (common->shapes[p_idx].isRectangle)
      {
        process = isPointInTriangle(mouse_coord, common->shapes[p_idx].point_a, common->shapes[p_idx].point_b, common->shapes[p_idx].point_c);
        process = process || isPointInTriangle(mouse_coord, common->shapes[p_idx].point_a, common->shapes[p_idx].point_d, common->shapes[p_idx].point_c);
      }

      __syncthreads();

      if (process || (common->shapes[p_idx].flag & 0x02))
      {
        common->shapes[p_idx].flag |= W_SHAPE_ROTATE_PROC;

        if (common->shapes[p_idx].isRectangle)
        {
          rectRotate(common->shapes + p_idx, mouse_shift.x * 0.2f, getCenterPoint(Vec2(common->shapes[p_idx].point_a), Vec2(common->shapes[p_idx].point_c)));

        }
        common->shapes[p_idx].angle += mouse_shift.x * 0.2f;
      }
    }
    else
    {
      common->shapes[p_idx].flag &= ~W_SHAPE_ROTATE_PROC;
    }

  }
}

void KernelRunner(dev::strct::ParticleData* particles,
  dev::strct::CommonData* common,
  dev::strct::EnviromentData* env_common,
  ball_data_s* balls,
  float* interval,
  int GeneratorNumber,
  int size)
{
  Debug::Message((std::wstring(L"Запуск ядра: \n\n") +
    std::wstring(L"Блоков : ") + std::to_wstring((size + dev::block_size - 1) / dev::block_size) + std::wstring(L"\n") +
    std::wstring(L"Потоков: ") + std::to_wstring(dev::block_size)
    ).c_str());

  cudaError_t cudaStatus;
  PhysicsKernel << < (size + dev::block_size - 1) / dev::block_size, dev::block_size >> > (particles, common, env_common, balls, interval, GeneratorNumber, size);
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    //cuda_error_code |= cuda_errors_enum::CE8;
    std::cout << (stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }
}

void RunCudaTest(Vec2 mouse_shift,
  Vec2 mouse_coord,
  int mouse_key,
  dev::strct::EnviromentData* common,
  int size)
{
  std::cout << "MOVE ---" << "\t" << "  butt: " << mouse_key << "\n";

  cudaError_t cudaStatus;
  TestKernel << < (size + dev::block_size - 1) / dev::block_size, dev::block_size >> > (mouse_shift, mouse_coord, mouse_key, common, size);
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    //cuda_error_code |= cuda_errors_enum::CE8;
    std::cout << (stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }
}

void ComputeCollision(dev::strct::EnviromentData* env_data,
  ball_data_s* balls,
  float* interval,
  int size)
{

  cudaError_t cudaStatus;
  CollisionKernel << < (size + dev::block_size - 1) / dev::block_size, dev::block_size >> > (env_data, balls, interval, size);
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    //cuda_error_code |= cuda_errors_enum::CE8;
    std::cout << (stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }
}

int wmain(int argc, wchar_t** argv) {

  std::wstring tmp(argv[0]);

  tmp = tmp.substr(0, tmp.find_last_of('\\'));
  tmp = tmp.substr(0, tmp.find_last_of('\\'));

  gl_context::Shader::locationDirectory = std::wstring(tmp) + std::wstring(SHADER_DIR);
  return gl_main();
}