#include <set>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"
#include <stdio.h>
#include "math_base.cuh"
#include "rawMesh.h"
#include "addConstraints.h"
#include "boostmap.h"
#include "fastDelaunay.h"

__global__ void fill1(int* a)
{
	int t = threadIdx.x + blockDim.x * blockIdx.x;
	a[t] = 1;
}

__device__ unsigned int rand16(unsigned int& seed, unsigned int tid)
{
	seed += tid;
	seed = (seed * seed + 10429) % 65536;
}

/**
c
|\ 
| \ 
| cp
|/
a--------b
*/
__global__ void see(double* points)
{
	JIO::Integer<16, false> t;
	t = 12;
	t = t * t;
	int a = t;
	printf("gpu %d\n", t);
}

#include <random>
void randomMap(double* points, int n, int seed)
{
	std::default_random_engine e;
	std::uniform_real_distribution<double> u(0.001, 0.999);
	e.seed(seed);

	points[0] = 0.75; points[1] = 0.5;
	points[2] = 0.75; points[3] = 0.2;
	points[4] = 0.25; points[5] = 0.2;
	points[6] = 0.5; points[7] = 0.75;
	
	for (int i = 8; i < n * 2; i++)
	{
		points[i] = u(e);
	}
}

void twoLineMap(double* points, int n, int seed)
{
	std::default_random_engine e;
	std::uniform_real_distribution<double> u(0.001, 0.999);
	e.seed(seed);

	//points[0] = 0.75; points[1] = 0.5;
	//points[2] = 0.75; points[3] = 0.2;
	//points[4] = 0.25; points[5] = 0.2;
	//points[6] = 0.5; points[7] = 0.75;

	for (int i = 0; i < n; i++)
	{
		const float2 L[2][2] = {
			{ { 0.0, 0.0 }, { 0.3, 0.5 } },
			{ { 0.7, 0.5 }, { 1.0, 1.0 } } };

		const int l = (u(e) < 0.5) ? 0 : 1;
		const float t = u(e);  // [ 0, 1 ]

		float x = (L[l][1].x - L[l][0].x) * t + L[l][0].x;
		float y = (L[l][1].y - L[l][0].y) * t + L[l][0].y;
		points[i * 2] = x;
		points[i * 2 + 1] = y;
	}
}

void squareMap(double* points, int n, int seed)
{
	double l = 0.05, r = 0.95;
	
	int p = sqrt(n);

	double d = (r - l) / (p - 1);

	std::default_random_engine e;
	std::uniform_real_distribution<double> u(-d / 3, d / 3);
	e.seed(seed);

	std::vector<double> X, Y;
	X.resize(p);
	Y.resize(p);
	for (int i = 0; i < p; i++)
	{
		X[i] = l + d * i;
	}
	for (int i = 0; i < p; i++)
	{
		Y[i] = l + d * i;
	}
	int tt = 0;
	for (int i = 0; i < p; i++)
	{
		for (int j = 0; j < p; j++)
		{
			points[tt * 2] = X[i];
			points[tt * 2 + 1] = Y[j];
			tt++;
		}
	}
}

void gaussianMap(double* points, int n, int seed)
{
	std::default_random_engine e;
	std::uniform_real_distribution<double> u(0.001, 0.999);
	e.seed(seed);

	for (int i = 0; i < n; i++)
	{
		double x1, x2, w;
		double tx, ty;

		do {
			do {
				x1 = 2.0 * u(e) - 1.0;
				x2 = 2.0 * u(e) - 1.0;
				w = x1 * x1 + x2 * x2;
			} while (w >= 1.0);

			w = sqrt((-2.0 * log(w)) / w);
			tx = x1 * w;
			ty = x2 * w;
		} while (tx < -3 || tx >= 3 || ty < -3 || ty >= 3);

		points[i*2] = 0.001 + (0.998) * ((tx + 3.0) / 6.0);
		points[i*2+1] = 0.001 + (0.998) * ((ty + 3.0) / 6.0);
	}
}
void thinCircleMap(double* points, int n, int seed)
{
	std::default_random_engine e;
	std::uniform_real_distribution<double> u(0.001, 0.999);
	e.seed(seed);

	for (int i = 0; i < n; i++)
	{
		double d, a;

		d = u(e) * 0.01;
		a = u(e) * 3.141592654 * 2;

		double x = (0.45 + d) * cos(a);
		double y = (0.45 + d) * sin(a);

		x += 0.5;
		y += 0.5;
		points[i * 2] = x;
		points[i * 2 + 1] = y;
	}
}
__global__ void transPoints(Point2d* points, double* fpoints, long long res, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	double fpx = fpoints[tid * 2];
	double fpy = fpoints[tid * 2 + 1];
	Point2d ret;
	long long x = fpx * res;
	if (x >= res) x = res - 1;
	if (x < 0) x = 0;
	long long y = fpy * res;
	if (y >= res) y = res - 1;
	if (y < 0) y = 0;

	ret.x = x;
	ret.y = y;
	points[tid] = ret;
}
__global__ void makePointAdj_step1(int3* tris, int3* orders, int* count, int* sons, int numT)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numT) return;
	if (sons[tid] == -1)
	{
		int3 t = tris[tid];
		int3 order;
		order.x = atomicAdd(&count[t.x], 1);
		order.y = atomicAdd(&count[t.y], 1);
		order.z = atomicAdd(&count[t.z], 1);
		orders[tid] = order;
	}
}

__global__ void makePointAdj_step2(int3* tris, int3* orders, int* offset, int* sons, int* adjTriIds, int numT)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numT) return;
	if (sons[tid] == -1)
	{
		int3 t = tris[tid];
		int3 order = orders[tid];
		adjTriIds[offset[t.x] + order.x] = tid;
		adjTriIds[offset[t.y] + order.y] = tid;
		adjTriIds[offset[t.z] + order.z] = tid;
	}
}

#include <cub/cub.cuh>


__device__ void hashEdge(int id, int x, int y, int* hashCount, int3* hashValue)
{
	int key = (x ^ y);
	int a = atomicAdd(hashCount + key, 1);
	if (a == 0)
	{
		hashValue[key] = make_int3(x, y, id);
	}
}

__global__ void clearAdjTris(int3* adjTris, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numT) return;

	adjTris[tid] = make_int3(-2, -2, -2);
}

__global__ void remakeAdjFace_s1(int3* tris, int3* adjTris, int* hashCount, int3* hashValue, int* mark, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numT) return;

	int3 t = tris[tid];
	int3 at = adjTris[tid];
	if (at.z == -2) {
		hashEdge(tid, t.x, t.y, hashCount, hashValue);
		mark[0] = 1;
	}
	if (at.x == -2) {
		hashEdge(tid, t.y, t.z, hashCount, hashValue);
		mark[0] = 1;
	}
	if (at.y == -2) {
		hashEdge(tid, t.z, t.x, hashCount, hashValue);
		mark[0] = 1;
	}
}

__global__ void remakeAdjFace_s2(int3* tris, int3* adjTris, int* hashCount, int3* hashValue, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numT) return;

	int3 t = tris[tid];
	int3 value = hashValue[t.x ^ t.y];

	if (((value.x == t.x && value.y == t.y) || (value.y == t.x && value.x == t.y)) && value.z != tid) {
		adjTris[tid].z = value.z;
		int3 tt = tris[value.z];
		int op = (tt.x ^ tt.y ^ tt.z ^ value.x ^ value.y);
		if (op == tt.x)
		{
			adjTris[value.z].x = tid;
		}
		if (op == tt.y)
		{
			adjTris[value.z].y = tid;
		}
		if (op == tt.z)
		{
			adjTris[value.z].z = tid;
		}
	}

	value = hashValue[t.y ^ t.z];
	if (((value.x == t.z && value.y == t.y) || (value.y == t.z && value.x == t.y)) && value.z != tid) {
		adjTris[tid].x = value.z;
		int3 tt = tris[value.z];
		int op = (tt.x ^ tt.y ^ tt.z ^ value.x ^ value.y);
		if (op == tt.x)
		{
			adjTris[value.z].x = tid;
		}
		if (op == tt.y)
		{
			adjTris[value.z].y = tid;
		}
		if (op == tt.z)
		{
			adjTris[value.z].z = tid;
		}
	}

	value = hashValue[t.x ^ t.z];
	if (((value.x == t.z && value.y == t.x) || (value.y == t.z && value.x == t.x)) && value.z != tid) {
		adjTris[tid].y = value.z;
		int3 tt = tris[value.z];
		int op = (tt.x ^ tt.y ^ tt.z ^ value.x ^ value.y);
		if (op == tt.x)
		{
			adjTris[value.z].x = tid;
		}
		if (op == tt.y)
		{
			adjTris[value.z].y = tid;
		}
		if (op == tt.z)
		{
			adjTris[value.z].z = tid;
		}
	}
}

__global__ void remakeAdjFace_s3(int3* tris, int3* adjTris, int* hashCount, int3* hashValue, int numH)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numH) return;

	if (hashCount[tid] != 0)
	{
		int3 value = hashValue[tid];
		int3 t = tris[value.z];
		if (t.x != value.x && t.x != value.y && adjTris[value.z].x == -2)
		{
			adjTris[value.z].x = -1;
		}
		if (t.y != value.x && t.y != value.y && adjTris[value.z].y == -2)
		{
			adjTris[value.z].y = -1;
		}
		if (t.z != value.x && t.z != value.y && adjTris[value.z].z == -2)
		{
			adjTris[value.z].z = -1;
		}
		hashCount[tid] = 0;
	}
}
__device__ void rand(unsigned int& seed)
{
	seed = (seed * 97 + 100519) % 2147483647;
}

__global__ void initSeed(unsigned int* seeds, int numR)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numR) return;
	unsigned int seed = tid;
	rand(seed);
	rand(seed);
	rand(seed);
	rand(seed);
	seeds[tid] = seed;
}

__global__ void testInCircle_s1(Point2d* points, int3* face2face, int3* face2Vertex, int* needFlip, int* needTest, int* atomicFace, unsigned int* seeds, int* d_mark, int numF)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numF * 3) return;

	unsigned int seed = seeds[tid];
	rand(seed);
	seeds[tid] = seed;
	int fid = tid % numF;
	int3 f2f = face2face[fid];
	if (needTest[tid] > 0) {
		int3 f2v = face2Vertex[fid];
		
		int ov = -1;
		if (tid < numF) { //x
			if (f2f.x < fid && f2f.x != -1) {
				int3 of2v = face2Vertex[f2f.x];
				ov = (of2v.x ^ of2v.y ^ of2v.z ^ f2v.y ^ f2v.z);
			}
		}
		else if (tid >= numF && tid < numF * 2) { //y
			if (f2f.y < fid && f2f.y != -1) {
				int3 of2v = face2Vertex[f2f.y];
				ov = (of2v.x ^ of2v.y ^ of2v.z ^ f2v.z ^ f2v.x);
			}
		}
		else { //z
			if (f2f.z < fid && f2f.z != -1) {
				int3 of2v = face2Vertex[f2f.z];
				ov = (of2v.x ^ of2v.y ^ of2v.z ^ f2v.x ^ f2v.y);
			}
		}

		int p = -1;
		if (ov != -1)
			p = inCircle(points[f2v.x], points[f2v.y], points[f2v.z], points[ov]);

		needFlip[tid] = p > 0 ? 1 : 0;
		needTest[tid] = 0;
	}

	if (needFlip[tid] == 1)
	{
		d_mark[0] = 1;
		if (tid < numF) { //x
			atomicMax(atomicFace + f2f.x, seed);
			atomicMax(atomicFace + fid, seed);
		}
		else if (tid >= numF && tid < numF * 2) { //y		
			atomicMax(atomicFace + f2f.y, seed);
			atomicMax(atomicFace + fid, seed);
		}
		else if (tid >= numF * 2 && tid < numF * 3) { //z
			atomicMax(atomicFace + f2f.z, seed);
			atomicMax(atomicFace + fid, seed);
		}
	}
}
/*
__global__ void testInCircle_s1_e_c(Point2d* points, int3* face2Vertex, int2* edge2face, int*needTest, int* needFlip, int* offset, int* atomicFace, int* d_mark, int numE, int numNT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numNT) return;
	int eid = LBS(offset, 0, numE, tid);

	int2 e2f = edge2face[eid];
	int p = -1;
	if (e2f.x != -1 && e2f.y != -1)
	{
		int3 f1 = face2Vertex[e2f.x];
		int3 f2 = face2Vertex[e2f.y];
		int ov;
		if (!contains(f1, f2.x))
		{
			ov = f2.x;
		}
		else if (!contains(f1, f2.y))
		{
			ov = f2.y;
		}
		else {
			ov = f2.z;
		}
		p = inCircle(points[f1.x], points[f1.y], points[f1.z], points[ov]);
	}
	int nf = p > 0 ? 1 : 0;
	
	needTest[eid] = 0;
	if (nf == 1)
	{
		d_mark[0] = 1;
		atomicMax(atomicFace + e2f.x, eid);
		atomicMax(atomicFace + e2f.y, eid);
	}
	needFlip[eid] = nf;
}
*/
__global__ void testInCircle_s1_e(Point2d* points, int3* face2Vertex, int2* edge2face, int* needFlip, int* needTest, int* atomicFace, unsigned int* seeds, int* d_mark, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	
	int2 e2f = edge2face[tid];
	if (needTest[tid] > 0) {
		int p = -1;
		if (e2f.x != -1 && e2f.y != -1)
		{
			int3 f1 = face2Vertex[e2f.x];
			int3 f2 = face2Vertex[e2f.y];
			int ov;
			if (!contains(f1, f2.x))
			{
				ov = f2.x;
			}
			else if (!contains(f1, f2.y))
			{
				ov = f2.y;
			}
			else {
				ov = f2.z;
			}
			p = inCircle(points[f1.x], points[f1.y], points[f1.z], points[ov]);
			//p = 1;
		}

		needFlip[tid] = p > 0 ? 1 : 0;
		needTest[tid] = 0;
	}

	if (needFlip[tid] == 1)
	{
		d_mark[0] = 1;
		atomicMax(atomicFace + e2f.x, tid);
		atomicMax(atomicFace + e2f.y, tid);
	}
}

__device__ void updateFace2Face(int3& f2f, int x, int y)
{
	if (f2f.x == x) f2f.x = y;
	else if (f2f.y == x) f2f.y = y;
	else if (f2f.z == x) f2f.z = y;
}
__global__ void testInCircle_s2(int3* face2face, int* needFlip, int* canFlip, int* atomicFace, unsigned int* seeds, int numF)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numF * 3) return;

	unsigned int seed = seeds[tid];

	if (needFlip[tid] == 1) {
		int fid = tid % numF;
		int3 f2f = face2face[fid];
		int tt = -1;
		int can;
		if (tid < numF) { //x
			if (atomicFace[f2f.x] != seed || atomicFace[fid] != seed) {
				can = 0;
			}
			else {
				can = 1;
			}
		}
		else if (tid >= numF && tid < numF * 2) { //y		
			if (atomicFace[f2f.y] != seed || atomicFace[fid] != seed) {
				can = 0;
			}
			else {
				can = 1;
			}
		}
		else { //z
			if (atomicFace[f2f.z] != seed || atomicFace[fid] != seed) {
				can = 0;
			}
			else {
				can = 1;
			}
		}
		if (can == 1) needFlip[tid] = 0;
		canFlip[tid] = can;
	}
	else {
		canFlip[tid] = 0;
	}
}

__device__ int getOtherEdgeId(int3* tris, int oid, int x, int y, int numF)
{
	int3 t = tris[oid];
	int p = t.x ^ t.y ^ t.z ^ x ^ y;
	if (p == t.x)
	{
		return oid;
	}
	else if (p == t.y)
	{
		return oid + numF;
	}
	else {
		return oid + numF * 2;
	}
}
__global__ void flipEdge(int3* face2Vertex, int3* face2face, int* canFlip, int* needTest, int numF)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numF) return;
	//if (tid != 8) return;
	//return;
	int f1, f2, f3, f4;
	int p1, p2, p3, p4;
	int3 f = face2face[tid];
	int3 v = face2Vertex[tid];
	int tt = -1;

	if (canFlip[tid] != 0) //flip X
	{
		//printf("flipd %d X\n", tid);
		tt = f.x;
		p1 = v.x;
		p2 = v.y;
		p4 = v.z;
		f1 = f.z;
		f2 = f.y;
	}

	if (canFlip[tid + numF] != 0) //flip Y
	{
		//printf("flipd %d Y\n", tid);
		tt = f.y;
		p1 = v.y;
		p2 = v.z;
		p4 = v.x;
		f1 = f.x;
		f2 = f.z;
	}

	if (canFlip[tid + numF * 2] != 0) //flip Z
	{
		//printf("flipd %d Z\n", tid);
		tt = f.z;
		p1 = v.z;
		p2 = v.x;
		p4 = v.y;
		f1 = f.y;
		f2 = f.x;
	}

	if (tt != -1) 
	{
		int3 of = face2face[tt];
		int3 ov = face2Vertex[tt];
		//if (tid == 9)
		//	printf("tt %d  of %d %d %d  ov %d %d %d\n", tt, of.x, of.y, of.z, ov.x, ov.y, ov.z);
		//if (tt == 9)
		//{
		//	printf("tid %d  of %d %d %d  ov %d %d %d\n", tid, of.x, of.y, of.z, ov.x, ov.y, ov.z);
		//}
		if (of.x == tid)
		{
			f3 = of.y; f4 = of.z;
			p3 = ov.x;
		}
		else if (of.y == tid)
		{
			f3 = of.z; f4 = of.x;
			p3 = ov.y;
		}
		else
		{
			f3 = of.x; f4 = of.y;
			p3 = ov.z;
		}

		int3 f2f;
		int3 f2v;

		f2v.x = p2;
		f2v.y = p3;
		f2v.z = p1;
		f2f.x = tt;
		f2f.y = f1;
		f2f.z = f3;
		face2face[tid] = f2f;
		face2Vertex[tid] = f2v;
		//if (tid == 24) printf("set 24\n");
		//if (tid == 9)
		//{
		//	printf("set tid %d %d %d %d %d %d\n", f2f.x, f2f.y, f2f.z, f2v.x, f2v.y, f2v.z);
		//}
		f2v.x = p4;
		f2v.y = p1;
		f2v.z = p3;
		f2f.x = tid;
		f2f.y = f4;
		f2f.z = f2;
		face2face[tt] = f2f;
		face2Vertex[tt] = f2v;
		//if (tt == 9)
		//{
		//	printf("set tt %d %d %d %d %d %d\n", f2f.x, f2f.y, f2f.z, f2v.x, f2v.y, f2v.z);
		//}
	}
}

__global__ void flipEdge_heal(int3* face2Vertex, int3* face2face, int numF)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numF) return;

	int3 f = face2face[tid];
	int3 v = face2Vertex[tid];
	
	int tt = f.y;
	if (tt != -1) {
		int3 ov = face2Vertex[tt];
		if (!contains(ov, v.x) || !contains(ov, v.z))
		{
			face2face[tid].y = face2face[tt].x;
		}
	}
	tt = f.z;
	if (tt != -1) {
		int3 ov = face2Vertex[tt];
		if (!contains(ov, v.x) || !contains(ov, v.y))
		{
			face2face[tid].z = face2face[tt].x;
		}
	}

	tt = f.x;
	if (tt != -1) {
		int3 ov = face2Vertex[tt];
		if (!contains(ov, v.y) || !contains(ov, v.z))
		{
			face2face[tid].x = face2face[tt].x;
		}
	}
}

__global__ void flipEdge_needTest(int3* face2Vertex, int3* face2face, int* canFlip, int* needTest, int numF)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numF) return;

	int3 f = face2face[tid];
	int3 v = face2Vertex[tid];

	if (canFlip[tid] != 0 || canFlip[tid + numF] != 0 || canFlip[tid + numF * 2] != 0) //flip X
	{
		//printf("flipd %d X\n", tid);
		needTest[tid] = 1;
		needTest[tid + numF] = 1;
		needTest[tid + numF * 2] = 1;
		if (f.y != -1)
			needTest[getOtherEdgeId(face2Vertex, f.y, v.x, v.z, numF)] = 1;
		if (f.z != -1)
			needTest[getOtherEdgeId(face2Vertex, f.z, v.x, v.y, numF)] = 1;

		int3 of = face2face[f.x];
		int3 ov = face2Vertex[f.x];

		needTest[f.x] = 1;
		needTest[f.x + numF] = 1;
		needTest[f.x + numF * 2] = 1;
		if (of.y != -1)
			needTest[getOtherEdgeId(face2Vertex, of.y, ov.z, ov.x, numF)] = 1;
		if (of.z != -1)
			needTest[getOtherEdgeId(face2Vertex, of.z, ov.x, ov.y, numF)] = 1;
	}
}
__global__ void flipEdgeDebug(int3* face2Vertex, int3* face2face, int* canFlip, int numF)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numF) return;
	//if (tid != 8) return;
	//return;
	int f1, f2, f3, f4;
	int p1, p2, p3, p4;
	int3 f = face2face[tid];
	int3 v = face2Vertex[tid];
	int tt = -1;
	
	if (canFlip[tid] != 0) //flip X
	{
		if (tid == 22)
			printf("flipd %d X\n", tid);
		tt = f.x;
		p1 = v.x;
		p2 = v.y;
		p4 = v.z;
		f1 = f.z;
		f2 = f.y;
	}

	if (canFlip[tid + numF] != 0) //flip Y
	{
		if (tid == 22) printf("flipd %d Y\n", tid);
		tt = f.y;
		p1 = v.y;
		p2 = v.z;
		p4 = v.x;
		f1 = f.x;
		f2 = f.z;
	}

	if (canFlip[tid + numF * 2] != 0) //flip Z
	{
		if (tid == 22) printf("flipd %d Z\n", tid);
		tt = f.z;
		p1 = v.z;
		p2 = v.x;
		p4 = v.y;
		f1 = f.y;
		f2 = f.x;
	}

	if (tid == 22)
		printf("f %d %d %d  v %d %d %d\n", f.x, f.y, f.z, v.x, v.y, v.z);
	if (tt != -1)
	{
		int3 of = face2face[tt];
		int3 ov = face2Vertex[tt];
		if (tid == 22) printf("%d  of %d %d %d  ov %d %d %d\n", tt, of.x, of.y, of.z, ov.x, ov.y, ov.z);
		if (of.x == tid)
		{
			f3 = of.y; f4 = of.z;
			p3 = ov.x;
		}
		else if (of.y == tid)
		{
			f3 = of.z; f4 = of.x;
			p3 = ov.y;
		}
		else
		{
			f3 = of.x; f4 = of.y;
			p3 = ov.z;
		}

		int3 f2f;
		int3 f2v;
		f2v.x = p2;
		f2v.y = p3;
		f2v.z = p1;
		f2f.x = tt;
		f2f.y = f1;
		f2f.z = f3;
		face2face[tid] = f2f;
		face2Vertex[tid] = f2v;
		if (tid == 22) printf("ret v %d %d %d\n", f2v.x, f2v.y, f2v.z);
		f2v.x = p4;
		f2v.y = p1;
		f2v.z = p3;
		f2f.x = tid;
		f2f.y = f4;
		f2f.z = f2;
		face2face[tt] = f2f;
		face2Vertex[tt] = f2v;
		if (tid == 22) printf("ret ov %d %d %d\n", f2v.x, f2v.y, f2v.z);
	}
}

__global__ void convertSample(int3* tris, int* sampleId, int numSamples, int numPoints, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	int3 t = tris[tid];
	if (t.x >= numSamples)
	{
		t.x = t.x - numSamples + numPoints;
	}
	else {
		t.x = sampleId[t.x];
	}
	if (t.y >= numSamples)
	{
		t.y = t.y - numSamples + numPoints;
	}
	else {
		t.y = sampleId[t.y];
	}
	if (t.z >= numSamples)
	{
		t.z = t.z - numSamples + numPoints;
	}
	else {
		t.z = sampleId[t.z];
	}
	tris[tid] = t;
}

__global__ void filterTriangles_s1(int* sons, int* count, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	if (sons[tid] == -1) count[tid] = 1; else count[tid] = 0;
}

__global__ void filterTriangles_s2(int3* tris, int3* newTris, int* sons, int* offset, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	if (sons[tid] == -1) {
		newTris[offset[tid]] = tris[tid];
	}
}

__global__ void filterTriangles_s2_new(int3* tris, int3* newTris, int3* adjTris, int3* newAdjTris, int* sons, int* offset, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	if (sons[tid] == -1) {
		newTris[offset[tid]] = tris[tid];
		int3 t = adjTris[tid];
		if (t.x != -1)
			t.x = offset[t.x];
		if (t.y != -1)
			t.y = offset[t.y];
		if (t.z != -1)
			t.z = offset[t.z];
		newAdjTris[offset[tid]] = t;
	}
}

void delaunay(Point2d* points, int3* tris, int3* adjTris, int numPoints, int numTriangles)
{
	unsigned int* randSeeds;
	utils::malloc(randSeeds, numTriangles * 3);
	utils::kernel(initSeed, numTriangles * 3, 128, randSeeds, numTriangles * 3);

	int* canFlip;
	utils::malloc(canFlip, numTriangles * 3);
	int* needFlip;
	utils::malloc(needFlip, numTriangles * 3);
	int* atomicFace;
	utils::malloc(atomicFace, numTriangles);
	int* needTest;
	utils::malloc(needTest, numTriangles * 3);
	int* d_mark;
	utils::malloc(d_mark, 1);
	int mark = 1;
	int time = 0;

	{
		cudaEvent_t start, stop;
		float elapsedTime = 0.0;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		utils::memset(needTest, numTriangles * 3, 1);
		while (true) {
			time++;
			//printf("time %d\n", time);
			utils::memset(atomicFace, numTriangles);
			utils::memset(d_mark, 1);
			utils::kernel(testInCircle_s1, numTriangles * 3, 128, points, adjTris, tris, needFlip, needTest, atomicFace, randSeeds, d_mark, numTriangles);
			mark = utils::getValue(d_mark, 0);
			if (mark == 0) { printf("flip times: %d\n", time - 1); break; }
			
			utils::kernel(testInCircle_s2, numTriangles * 3, 128, adjTris, needFlip, canFlip, atomicFace, randSeeds, numTriangles);
			utils::kernel(flipEdge, numTriangles, 128, tris, adjTris, canFlip, needTest, numTriangles);
			utils::kernel(flipEdge_heal, numTriangles, 128, tris, adjTris, numTriangles);
			utils::kernel(flipEdge_needTest, numTriangles, 128, tris, adjTris, canFlip, needTest, numTriangles);
			//draw(points, tris, canFlip, needTest, numPoints, numTriangles);
			//cudaDeviceSynchronize();
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		std::cout << "flipEdge" << " time: " << elapsedTime << " ms\n";
	}

	utils::release(randSeeds, numTriangles * 3);
	utils::release(canFlip, numTriangles * 3);
	utils::release(needFlip, numTriangles * 3);
	utils::release(atomicFace, numTriangles);
	utils::release(d_mark, 1);
}

__global__ void testInCircle_s2_e(int2* edge2face, int* needFlip, int* canFlip, int* atomicFace, int* futureFace, unsigned int* seeds, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	if (needFlip[tid] == 1) {
		int2 e2f = edge2face[tid];
		int can;
		if (atomicFace[e2f.x] != tid || atomicFace[e2f.y] != tid)
		{
			can = 0;
		}
		else {
			can = 1;
		}
		
		if (can == 1) {
			needFlip[tid] = 0;
			futureFace[e2f.x] = e2f.y;
			futureFace[e2f.y] = e2f.x;
		}
		canFlip[tid] = can;
	}
	else {
		canFlip[tid] = 0;
	}
}

__global__ void flipEdge_e(int2* edge2face, int2* edges, int3* face2Vertex, int* canFlip, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	//if (tid != 8) return;
	//return;
	int p1, p2, p3, p4;

	if (canFlip[tid] != 0)
	{
		int2 e2f = edge2face[tid];
		int2 e2v = edges[tid];
		int3 fv1 = face2Vertex[e2f.x];
		int3 fv2 = face2Vertex[e2f.y];
		p1 = fv1.x ^ fv1.y ^ fv1.z ^ e2v.x ^ e2v.y;
		p3 = fv2.x ^ fv2.y ^ fv2.z ^ e2v.x ^ e2v.y;
		if (p1 == fv1.x)
		{
			p2 = fv1.y; p4 = fv1.z;
		}
		else if (p1 == fv1.y){
			p2 = fv1.z; p4 = fv1.x;
		}
		else {
			p2 = fv1.x; p4 = fv1.y;
		}

		edges[tid] = make_int2(p1, p3);
		int3 f2v;

		f2v.x = p2;
		f2v.y = p3;
		f2v.z = p1;
		face2Vertex[e2f.x] = f2v;

		f2v.x = p4;
		f2v.y = p1;
		f2v.z = p3;
		face2Vertex[e2f.y] = f2v;
	}
}

__global__ void flipEdge_heal_e(int3* face2Vertex, int2* edges, int2* edge2face, int* futureFace, int* needTest, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 ev = edges[tid];
	int2 e2f = edge2face[tid];

	needTest[tid] = 0;
	if (e2f.x != -1)
	{
		int ff = futureFace[e2f.x];
		if (ff != -1) {
			if (ff != e2f.y) {
				needTest[tid] = 1;
			}
			int3 f = face2Vertex[e2f.x];
			if (!contains(f, ev.x) || !contains(f, ev.y))
			{
				edge2face[tid].x = ff;
			}
		}
	}

	if (e2f.y != -1)
	{
		int ff = futureFace[e2f.y];
		if (ff != -1) {
			if (ff != e2f.x) {
				needTest[tid] = 1;
			}
			int3 f = face2Vertex[e2f.y];
			if (!contains(f, ev.x) || !contains(f, ev.y))
			{
				edge2face[tid].y = ff;
			}
		}
	}
}

__global__ void remakeAdjFaceSort_s1(int3* tris, int2* edges, int* edgesValue, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;

	int3 t = tris[tid];
	if (t.x < t.y)
		edges[tid * 3] = make_int2(t.x, t.y);
	else
		edges[tid * 3] = make_int2(t.y, t.x);
	edgesValue[tid * 3] = tid * 3;

	if (t.y < t.z)
		edges[tid * 3 + 1] = make_int2(t.y, t.z);
	else
		edges[tid * 3 + 1] = make_int2(t.z, t.y);
	edgesValue[tid * 3 + 1] = tid * 3 + 1;

	if (t.z < t.x)
		edges[tid * 3 + 2] = make_int2(t.z, t.x);
	else
		edges[tid * 3 + 2] = make_int2(t.x, t.z);
	edgesValue[tid * 3 + 2] = tid * 3 + 2;
}

__global__ void remakeAdjFaceSort_s2(int2* edges, int* edgesValue, int3* adjTris, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE - 1) return;

	int2 e = edges[tid];
	int2 o = edges[tid + 1];
	if (e.x == o.x && e.y == o.y)
	{
		int ez = edgesValue[tid];
		int oz = edgesValue[tid + 1];

		if (ez % 3 == 0) {
			adjTris[ez / 3].z = oz / 3;
		}
		else if (ez % 3 == 1) {
			adjTris[ez / 3].x = oz / 3;
		}
		else {
			adjTris[ez / 3].y = oz / 3;
		}

		if (oz % 3 == 0) {
			adjTris[oz / 3].z = ez / 3;
		}
		else if (oz % 3 == 1) {
			adjTris[oz / 3].x = ez / 3;
		}
		else {
			adjTris[oz / 3].y = ez / 3;
		}
	}
}
void updateAdjSort(void* cubTemp, size_t cubBytes, int3* tris, int3* adjTris, int numPoints, int numTriangles)
{
	int2* edges;
	int* edgesValue;
	utils::malloc(edges, numTriangles * 3);
	utils::malloc(edgesValue, numTriangles * 3);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::memset(adjTris, numTriangles, -1);
	utils::kernel(remakeAdjFaceSort_s1, numTriangles, 128, tris, edges, edgesValue, numTriangles);

	utils::mergeSort(cubTemp, cubBytes, edges, edgesValue, numTriangles * 3);
	//utils::sort(cubTemp, cubBytes, edges, edges, edgesValue, edgesValue, numTriangles * 3);

	utils::kernel(remakeAdjFaceSort_s2, numTriangles * 3, 128, edges, edgesValue, adjTris, numTriangles * 3);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "update adj sort" << " time: " << elapsedTime << " ms\n";
}

__global__ void remakeAdjFaceRadixSort_s1(int3* tris, int* edgesKey, int3* edgesValue, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;

	int3 t = tris[tid];
	int3 value;
	if (t.x < t.y) {
		edgesKey[tid * 3] = (t.x << 4) ^ t.y;
		value.x = t.x; value.y = t.y;
		value.z = tid * 3;
	}
	else {
		edgesKey[tid * 3] = (t.y << 4) ^ t.x;
		value.x = t.y; value.y = t.x;
		value.z = tid * 3;
	}
	edgesValue[tid * 3] = value;

	if (t.y < t.z) {
		edgesKey[tid * 3 + 1] = (t.y << 4) ^ t.z;
		value.x = t.y; value.y = t.z;
		value.z = tid * 3 + 1;
	}
	else {
		edgesKey[tid * 3 + 1] = (t.z << 4) ^ t.y;
		value.x = t.z; value.y = t.y;
		value.z = tid * 3 + 1;
	}
	edgesValue[tid * 3 + 1] = value;

	if (t.x < t.z) {
		edgesKey[tid * 3 + 2] = (t.x << 4) ^ t.z;
		value.x = t.x; value.y = t.z;
		value.z = tid * 3 + 2;
	}
	else {
		edgesKey[tid * 3 + 2] = (t.z << 4) ^ t.x;
		value.x = t.z; value.y = t.x;
		value.z = tid * 3 + 2;
	}
	edgesValue[tid * 3 + 2] = value;
}

__global__ void remakeAdjFaceRadixSort_s2(int* edges, int3* edgesValue, int3* adjTris, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE - 1) return;

	int hash = edges[tid];
	int3 value = edgesValue[tid];
	int p = tid + 1;
	while (p < numE && edges[p] == hash)
	{
		int3 ov = edgesValue[p];
		if (ov.x == value.x && ov.y == value.y)
		{
			int ez = value.z;
			int oz = ov.z;

			if (ez % 3 == 0) {
				adjTris[ez / 3].z = oz / 3;
			}
			else if (ez % 3 == 1) {
				adjTris[ez / 3].x = oz / 3;
			}
			else {
				adjTris[ez / 3].y = oz / 3;
			}

			if (oz % 3 == 0) {
				adjTris[oz / 3].z = ez / 3;
			}
			else if (oz % 3 == 1) {
				adjTris[oz / 3].x = ez / 3;
			}
			else {
				adjTris[oz / 3].y = ez / 3;
			}
		}
		p++;
	}
}
void updateAdjRadixSort(void* cubTemp, size_t cubBytes, int3* tris, int3* adjTris, int numPoints, int numTriangles)
{
	int* edgesKey, *edgesKeySorted;
	int3* edgesValue, *edgesValueSorted;
	utils::malloc(edgesKey, numTriangles * 3);
	utils::malloc(edgesKeySorted, numTriangles * 3);
	utils::malloc(edgesValue, numTriangles * 3);
	utils::malloc(edgesValueSorted, numTriangles * 3);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::memset(adjTris, numTriangles, -1);
	utils::kernel("s1", remakeAdjFaceRadixSort_s1, numTriangles, 128, tris, edgesKey, edgesValue, numTriangles);

	utils::sort(cubTemp, cubBytes, edgesKey, edgesKeySorted, edgesValue, edgesValueSorted, numTriangles * 3);
	//utils::sort(cubTemp, cubBytes, edges, edges, edgesValue, edgesValue, numTriangles * 3);

	utils::kernel("s2", remakeAdjFaceRadixSort_s2, numTriangles * 3, 128, edgesKeySorted, edgesValueSorted, adjTris, numTriangles * 3);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "update adj sort" << " time: " << elapsedTime << " ms\n";

	utils::release(edgesKey, numTriangles * 3);
	utils::release(edgesKeySorted, numTriangles * 3);
	utils::release(edgesValue, numTriangles * 3);
	utils::release(edgesValueSorted, numTriangles * 3);
}

__global__ void remakeEdgeRadixSort_s2(int* edges, int3* edgesValue, int* count, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int hash = edges[tid];
	int3 value = edgesValue[tid];

	int R = tid;
	while (R < numE && edges[R] == hash)
	{
		R++;
	}
	int L = tid;
	while (L >= 0 && edges[L] == hash)
	{
		L--;
	}
	L++;
	count[tid] = 0;
	for (int i = L; i < R; i++)
	{
		if (i == tid) continue;
		int3 v = edgesValue[i];
		if (v.x == value.x && v.y == value.y) {
			if (i < tid)
			{
				count[tid] = 1; break;	
			}
			else {
				count[tid] = 0; break;
			}
		}
	}
}

__global__ void remakeEdgeRadixSort_s3(int* edges, int3* edgesValue, int* offset, int2* edge2v, int2* edge2face, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int hash = edges[tid];
	int3 value = edgesValue[tid];

	int R = tid;
	while (R < numE && edges[R] == hash)
	{
		R++;
	}
	int L = tid;
	while (L >= 0 && edges[L] == hash)
	{
		L--;
	}
	L++;
	int count = 0;
	int ad = tid;
	for (int i = L; i < R; i++)
	{
		if (i == tid) continue;
		int3 v = edgesValue[i];
		if (v.x == value.x && v.y == value.y) {
			if (i < tid)
			{
				ad = i;
				count = 1; break;
			}
			else {
				count = 0; break;
			}
		}
	}
	ad = ad - offset[ad];
	if (count == 0)
	{
		edge2v[ad] = make_int2(value.x, value.y);
		edge2face[ad].x = value.z / 3;
	}
	else {
		edge2face[ad].y = value.z / 3;
	}
}

__global__ void reorderEdge_s1(Point2d* points, int2* edges, int* labels, int* codes, unsigned int logRes, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numE) return;

	int2 e = edges[tid];
	Point2d p1 = points[e.x];
	Point2d p2 = points[e.y];
	int x = (p1.x + p2.x) / 2;
	int y = (p1.y + p2.y) / 2;
	if (e.x < e.y) {
		x = p1.x;
		y = p1.y;
	}
	else
	{
		x = p2.x;
		y = p2.y;
	}
	x = x >> (logRes - 15);
	y = y >> (logRes - 15);

	int c = 0;
	for (int i = 0; i < 15; i++)
	{
		c |= ((x >> i) & 1) << (i * 2);
		c |= ((y >> i) & 1) << (i * 2 + 1);
	}
	labels[tid] = tid;
	codes[tid] = c;
}

__global__ void reorderEdge_s2(int2* edges, int2* edge2face, int2* edgesSorted, int2* edge2faceSorted, int* labelsSorted, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numE) return;

	int t = labelsSorted[tid];
	edgesSorted[tid] = edges[t];
	edge2faceSorted[tid] = edge2face[t];
}

void reorderEdge(void* cubTemp, size_t cubBytes, Point2d* points, int2*& edges, int2*& edge2face, int numTriangles, int numEdges)
{
	int* codes;
	int* labels;
	int* codesSorted, *labelsSorted;
	int2* edgesSorted, * edge2faceSorted;
	utils::malloc(codes, numEdges);
	utils::malloc(labels, numEdges);
	utils::malloc(codesSorted, numEdges);
	utils::malloc(labelsSorted, numEdges);
	utils::malloc(edgesSorted, numEdges);
	utils::malloc(edge2faceSorted, numEdges);
	
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::kernel(reorderEdge_s1, numEdges, 128, points, edges, labels, codes, logPointRes, numEdges);

	utils::sort(cubTemp, cubBytes, codes, codesSorted, labels, labelsSorted, numEdges);

	utils::kernel(reorderEdge_s2, numEdges, 128, edges, edge2face, edgesSorted, edge2faceSorted, labelsSorted, numEdges);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "reorder edges" << " time: " << elapsedTime << " ms\n";

	utils::release(edges, numTriangles * 2);
	utils::release(edge2face, numTriangles * 2);
	utils::release(codes, numEdges);
	utils::release(labels, numEdges);
	utils::release(codesSorted, numEdges);
	utils::release(labelsSorted, numEdges);

	edges = edgesSorted;
	edge2face = edge2faceSorted;
}
void updateEdgeRadixSort(void* cubTemp, size_t cubBytes, int3* tris, int2*& edges, int2*& edge2face, int& numEdges, int numPoints, int numTriangles)
{
	int* edgesKey, * edgesKeySorted;
	int3* edgesValue, * edgesValueSorted;
	int* count, *offset;
	utils::malloc(edgesKey, numTriangles * 3);
	utils::malloc(edgesKeySorted, numTriangles * 3);
	utils::malloc(edgesValue, numTriangles * 3);
	utils::malloc(edgesValueSorted, numTriangles * 3);
	utils::malloc(count, numTriangles * 3 + 1);
	utils::malloc(offset, numTriangles * 3 + 1);
	

	utils::kernel("s1", remakeAdjFaceRadixSort_s1, numTriangles, 128, tris, edgesKey, edgesValue, numTriangles);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::sort(cubTemp, cubBytes, edgesKey, edgesKeySorted, edgesValue, edgesValueSorted, numTriangles * 3);
	//utils::sort(cubTemp, cubBytes, edges, edges, edgesValue, edgesValue, numTriangles * 3);

	

	utils::kernel("s2", remakeEdgeRadixSort_s2, numTriangles * 3, 128, edgesKeySorted, edgesValueSorted, count, numTriangles * 3);
	utils::exlusiveScan(cubTemp, cubBytes, count, offset, numTriangles * 3 + 1);
	printf("numT %d\n", numTriangles);
	numEdges = numTriangles * 3 - utils::getValue(offset, numTriangles * 3);
	utils::malloc(edges, numEdges);
	utils::malloc(edge2face, numEdges);
	utils::kernel("s3", remakeEdgeRadixSort_s3, numTriangles * 3, 128, edgesKeySorted, edgesValueSorted, offset, edges, edge2face, numTriangles * 3);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "update edge" << " time: " << elapsedTime << " ms\n";

	utils::release(edgesKey, numTriangles * 3);
	utils::release(edgesKeySorted, numTriangles * 3);
	utils::release(edgesValue, numTriangles * 3);
	utils::release(edgesValueSorted, numTriangles * 3);
	utils::release(count, numTriangles * 3 + 1);
	utils::release(offset, numTriangles * 3 + 1);
}

__device__ int getX(const int3& t, int y)
{
	if (y % 3 == 0)
	{
		return t.x < t.y ? t.x : t.y;
	}
	if (y % 3 == 1)
	{
		return t.y < t.z ? t.y : t.z;
	}
	if (y % 3 == 2)
	{
		return t.x < t.z ? t.x : t.z;
	}
}
__global__ void remakeEdgeRadixSort_s1_new(int3* tris, int* edgesKey, int* edgesValue, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;

	int3 t = tris[tid];
	//int2 value;
	if (t.x < t.y) {
		edgesKey[tid * 3] = (t.x << 4) ^ t.y;
		//value.x = t.x; value.y = tid * 3;
	}
	else {
		edgesKey[tid * 3] = (t.y << 4) ^ t.x;
		//value.x = t.y; value.y = tid * 3;
	}
	//edgesValue[tid * 3] = value;
	edgesValue[tid * 3] = tid * 3;

	if (t.y < t.z) {
		edgesKey[tid * 3 + 1] = (t.y << 4) ^ t.z;
		//value.x = t.y; value.y = tid * 3 + 1;
	}
	else {
		edgesKey[tid * 3 + 1] = (t.z << 4) ^ t.y;
		//value.x = t.z; value.y = tid * 3 + 1;
	}
	//edgesValue[tid * 3 + 1] = value;
	edgesValue[tid * 3 + 1] = tid * 3 + 1;

	if (t.x < t.z) {
		edgesKey[tid * 3 + 2] = (t.x << 4) ^ t.z;
		//value.x = t.x; value.y = tid * 3 + 2;
	}
	else {
		edgesKey[tid * 3 + 2] = (t.z << 4) ^ t.x;
		//value.x = t.z; value.y = tid * 3 + 2;
	}
	//edgesValue[tid * 3 + 2] = value;
	edgesValue[tid * 3 + 2] = tid * 3 + 2;
}

__global__ void remakeEdgeRadixSort_s1_addC(int2* cons, int* edgesKey, int* edgesValue, int numT, int numC)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numC) return;
	int2 con = cons[tid];
	int ck;
	int cv;
	if (con.x < con.y) {
		ck = (con.x << 4) ^ con.y;
		cv = -(con.x + 1);
	}
	else {
		ck = (con.y << 4) ^ con.x;
		cv = -(con.y + 1);
	}
	edgesKey[numT * 3 + tid] = ck;
	edgesValue[numT * 3 + tid] = cv;
}
__global__ void remakeEdgeRadixSort_s2_new(int3* tris, int* edges, int* edgesValue, int* count, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int hash = edges[tid];
	int2 value;
	value.y = edgesValue[tid];
	if (value.y < 0) return;
	value.x = getX(tris[value.y / 3], value.y);

	int R = tid;
	while (R < numE && edges[R] == hash)
	{
		R++;
	}
	int L = tid;
	while (L >= 0 && edges[L] == hash)
	{
		L--;
	}
	L++;
	int c = 0;
	for (int i = L; i < R; i++)
	{
		if (i == tid) continue;
		int2 v;
		v.y = edgesValue[i];
		if (v.y < 0) continue;
		v.x = getX(tris[v.y / 3], v.y);
		if (v.x == value.x) {
			if (v.y < value.y)
			{
				c = 1; break;
			}
			else {
				c = 0; break;
			}
		}
	}
	count[value.y] = c;
}

__global__ void remakeEdgeRadixSort_s3_new(int3* tris, int* edges, int* edgesValue, int* offset, int2* edge2v, int2* edge2face, int* noFlips, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int hash = edges[tid];
	int2 value;
	value.y = edgesValue[tid];
	if (value.y < 0) return;
 	value.x = getX(tris[value.y / 3], value.y);

	int R = tid;
	while (R < numE && edges[R] == hash)
	{
		R++;
	}
	int L = tid;
	while (L >= 0 && edges[L] == hash)
	{
		L--;
	}
	L++;
	int count = 0;
	int ad = value.y;
	int noFlip = 0;
	for (int i = L; i < R; i++)
	{
		if (i == tid) continue;
		int2 v;
		v.y = edgesValue[i];
		if (v.y >= 0) {
			v.x = getX(tris[v.y / 3], v.y);
		}
		else {
			v.x = -v.y - 1;
		}
		if (v.x == value.x) {
			if (v.y < 0) {
				noFlip = 1;
			}
			else {
				if (v.y < value.y)
				{
					ad = v.y;
					count = 1;
				}
				else {
					count = 0;
				}
			}
		}
	}
	ad = ad - offset[ad];
	if (count == 0)
	{
		edge2v[ad] = make_int2(value.x, (hash ^ (value.x << 4)));
		edge2face[ad].x = value.y / 3;
		if (noFlip == 1) {
			noFlips[ad] = 1;
		}
	}
	else {
		edge2face[ad].y = value.y / 3;
	}
}


__global__ void remakeFaceRadixSort_s1(int3* tris, int* edgesKey, int2* edgesValue, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;

	int3 t = tris[tid];
	int2 value;
	if (t.x < t.y) {
		edgesKey[tid * 3] = (t.x << 4) ^ t.y;
		value.x = t.x; value.y = tid * 3;
	}
	else {
		edgesKey[tid * 3] = (t.y << 4) ^ t.x;
		value.x = t.y; value.y = tid * 3;
	}
	edgesValue[tid * 3] = value;

	if (t.y < t.z) {
		edgesKey[tid * 3 + 1] = (t.y << 4) ^ t.z;
		value.x = t.y; value.y = tid * 3 + 1;
	}
	else {
		edgesKey[tid * 3 + 1] = (t.z << 4) ^ t.y;
		value.x = t.z; value.y = tid * 3 + 1;
	}
	edgesValue[tid * 3 + 1] = value;

	if (t.x < t.z) {
		edgesKey[tid * 3 + 2] = (t.x << 4) ^ t.z;
		value.x = t.x; value.y = tid * 3 + 2;
	}
	else {
		edgesKey[tid * 3 + 2] = (t.z << 4) ^ t.x;
		value.x = t.z; value.y = tid * 3 + 2;
	}
	edgesValue[tid * 3 + 2] = value;
}

double updateEdgeRadixSort_new(void* cubTemp, size_t cubBytes, int3* tris, int2*& edges, int2*& edge2face, int2* cons, int*& noFlips, int numC, int& numEdges, int numPoints, int numTriangles)
{
	int* edgesKey, * edgesKeySorted;
	int* edgesValue, * edgesValueSorted;
	int* count;
	utils::malloc(edgesKey, numTriangles * 3 + numC);
	utils::malloc(edgesKeySorted, numTriangles * 3 + numC);
	utils::malloc(edgesValue, numTriangles * 3 + numC);
	utils::malloc(edgesValueSorted, numTriangles * 3 + numC);
	utils::malloc(count, numTriangles * 3 + 1 + numC);

	utils::malloc(edges, (int)numTriangles * 2);
	utils::malloc(edge2face, (int)numTriangles * 2);
	//utils::kernel("s1", remakeAdjFaceRadixSort_s1, numTriangles, 128, tris, edgesKey, edgesValue, numTriangles);
	utils::malloc(noFlips, (int)numTriangles * 2);
	
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::memset(count, numTriangles * 3 + 1 + numC);
	utils::kernel(remakeEdgeRadixSort_s1_new, numTriangles, 128, tris, edgesKey, edgesValue, numTriangles);
	if (numC != 0)
	{
		utils::kernel(remakeEdgeRadixSort_s1_addC, numC, 128, cons, edgesKey, edgesValue, numTriangles, numC);
	}
	utils::sort(cubTemp, cubBytes, edgesKey, edgesKeySorted, edgesValue, edgesValueSorted, numTriangles * 3 + numC);
	//utils::sort(cubTemp, cubBytes, edges, edges, edgesValue, edgesValue, numTriangles * 3);
	
	
	utils::kernel(remakeEdgeRadixSort_s2_new, numTriangles * 3 + numC, 128, tris, edgesKeySorted, edgesValueSorted, count, numTriangles * 3 + numC);
	//utils::kernel("s2", remakeEdgeRadixSort_s2, numTriangles * 3, 128, edgesKeySorted, edgesValueSorted, count, numTriangles * 3);
	utils::exlusiveScan(cubTemp, cubBytes, count, count, numTriangles * 3 + 1 + numC);
	//printf("%d\n", utils::getValue(count, numTriangles * 3 + numC));
	numEdges = numTriangles * 3 - utils::getValue(count, numTriangles * 3 + numC);

	utils::memset(edge2face, numEdges, -1);
	utils::memset(noFlips, (int)numTriangles * 2);
	utils::kernel(remakeEdgeRadixSort_s3_new, numTriangles * 3 + numC, 128, tris, edgesKeySorted, edgesValueSorted, count, edges, edge2face, noFlips, numTriangles * 3 + numC);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "update edge" << " time: " << elapsedTime << " ms\n";

	utils::release(edgesKey, numTriangles * 3 + numC);
	utils::release(edgesKeySorted, numTriangles * 3 + numC);
	utils::release(edgesValue, numTriangles * 3 + numC);
	utils::release(edgesValueSorted, numTriangles * 3 + numC);
	utils::release(count, numTriangles * 3 + 1 + numC);
	return elapsedTime;
}

__global__ void getFace2Edge(int3* tris, int2* edges, int2* edge2face, int3* face2edge, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 ef = edge2face[tid];
	int2 ev = edges[tid];
	bool swap = false;
	if (ef.x != -1)
	{
		int3 t = tris[ef.x];
		int xid = getIdInt3(t, ev.x);
		int yid = getIdInt3(t, ev.y);
		getInt3(face2edge + ef.x, 3 - xid - yid) = tid;
		if ((xid + 1) % 3 != yid)
		{
			swap = true;
		}
	}
	if (ef.y != -1)
	{
		int3 t = tris[ef.y];
		int xid = getIdInt3(t, ev.x);
		int yid = getIdInt3(t, ev.y);
		getInt3(face2edge + ef.y, 3 - xid - yid) = tid;
		if ((yid + 1) % 3 != xid)
		{
			swap = true;
		}
	}

	if (swap)
	{
		edges[tid] = { ev.y, ev.x };
	}
}

__global__ void getEdge2Edge(int3* face2edge, int2* edges, int2* edgeSide, int2* edge2face, int4* edge2edge, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 ef = edge2face[tid];
	int2 ev = edges[tid];
	int4 ret = {-1, -1, -1, -1};
	int2 side = { -1, -1 };
	if (ef.x != -1)
	{
		int3 fe = face2edge[ef.x];
		int eid = getIdInt3(fe, tid);
		ret.x = getInt3(&fe, (eid + 1) % 3);
		if (edges[ret.x].x != ev.y)
		{
			side.x = edges[ret.x].x;
			ret.x = ret.x * 2 + 1;
		}
		else {
			side.x = edges[ret.x].y;
			ret.x = ret.x * 2;
		}
		ret.y = getInt3(&fe, (eid + 2) % 3);
		if (edges[ret.y].x != ev.x)
		{
			ret.y = ret.y * 2;
		}
		else {
			ret.y = ret.y * 2 + 1;
		}
	}

	if (ef.y != -1)
	{
		int3 fe = face2edge[ef.y];
		int eid = getIdInt3(fe, tid);
		ret.z = getInt3(&fe, (eid + 1) % 3);
		if (edges[ret.z].x != ev.x)
		{
			side.y = edges[ret.z].x;
			ret.z = ret.z * 2 + 1;
		}
		else {
			side.y = edges[ret.z].y;
			ret.z = ret.z * 2;
		}
		ret.w = getInt3(&fe, (eid + 2) % 3);
		if (edges[ret.w].x != ev.y)
		{
			ret.w = ret.w * 2;
		}
		else {
			ret.w = ret.w * 2 + 1;
		}
	}
	
	edge2edge[tid] = ret;
	edgeSide[tid] = side;
}
double updateEdgeToEdge(void* cubTemp, size_t cubBytes, int3* tris, int2* edges, int2*& edgeSide, int2* edge2face, int4*& edge2edge, int numEdges, int numTriangles)
{
	int3* face2edge;
	utils::malloc(edge2edge, numEdges);
	utils::malloc(face2edge, numTriangles);
	utils::malloc(edgeSide, numEdges);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::kernel(getFace2Edge, numEdges, 128, tris, edges, edge2face, face2edge, numEdges);
	utils::kernel(getEdge2Edge, numEdges, 128, face2edge, edges, edgeSide, edge2face, edge2edge, numEdges);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "update edge2edge" << " time: " << elapsedTime << " ms\n";

	utils::release(face2edge, numTriangles);
	return elapsedTime;
}
__global__ void reorderFace_s1(Point2d* points, int3* tris, int* labels, int* codes, unsigned int logRes, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numT) return;

	int3 e = tris[tid];
	Point2d p1 = points[e.x];
	Point2d p2 = points[e.y];
	Point2d p3 = points[e.z];

	int x = ((p1.x >> (logRes - 15)) + 
		     (p2.x >> (logRes - 15)) +
		     (p3.x >> (logRes - 15))) / 3;
	int y = ((p1.y >> (logRes - 15)) +
		     (p2.y >> (logRes - 15)) +
		     (p3.y >> (logRes - 15))) / 3;

	int c = 0;
	for (int i = 0; i < 15; i++)
	{
		c |= ((x >> i) & 1) << (i * 2);
		c |= ((y >> i) & 1) << (i * 2 + 1);
	}
	labels[tid] = tid;
	codes[tid] = c;
}

__global__ void reorderFace_s2(int3* faces, int3* facesSorted, int* labelsSorted, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numT) return;

	int t = labelsSorted[tid];
	facesSorted[tid] = faces[t];
}

double reorderFaces(void* cubTemp, size_t cubBytes, Point2d* points, int3*& tris, int numTriangles)
{
	int* codes;
	int* labels;
	int* codesSorted, * labelsSorted;
	int3* facesSorted;
	utils::malloc(codes, numTriangles);
	utils::malloc(labels, numTriangles);
	utils::malloc(codesSorted, numTriangles);
	utils::malloc(labelsSorted, numTriangles);
	utils::malloc(facesSorted, numTriangles);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::kernel(reorderFace_s1, numTriangles, 128, points, tris, labels, codes, logPointRes, numTriangles);

	utils::sort(cubTemp, cubBytes, codes, codesSorted, labels, labelsSorted, numTriangles);

	utils::kernel(reorderFace_s2, numTriangles, 128, tris, facesSorted, labelsSorted, numTriangles);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "reorder faces" << " time: " << elapsedTime << " ms\n";

	utils::release(codes, numTriangles);
	utils::release(labels, numTriangles);
	utils::release(codesSorted, numTriangles);
	utils::release(labelsSorted, numTriangles);

	utils::release(tris, numTriangles);

	tris = facesSorted;
	return elapsedTime;
}

__global__ void reorderPoint_s1(Point2d* points, int* labels, int* codes, unsigned int logRes, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numP) return;

	Point2d p1 = points[tid];

	int x = p1.x >> (logRes - 15);
	int y = p1.y >> (logRes - 15);

	int c = 0;
	for (int i = 0; i < 15; i++)
	{
		c |= ((x >> i) & 1) << (i * 2);
		c |= ((y >> i) & 1) << (i * 2 + 1);
	}
	labels[tid] = tid;
	codes[tid] = c;
}

__global__ void reorderPoint_s2(Point2d* points, Point2d* pointsSorted, int* labels, int* labelsSorted, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numP) return;

	int t = labelsSorted[tid];
	pointsSorted[tid] = points[t];
	labels[t] = tid;
}

__global__ void reorderPoint_s3(int3* tris, int* labels, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numT) return;
	int3 t = tris[tid];
	t.x = labels[t.x];
	t.y = labels[t.y];
	t.z = labels[t.z];
	tris[tid] = t;
}

__global__ void reorderPoint_s4(int2* cons, int* labels, int numC)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numC) return;
	int2 t = cons[tid];
	t.x = labels[t.x];
	t.y = labels[t.y];
	cons[tid] = t;
}
double reorderPoints(void* cubTemp, size_t cubBytes, Point2d*& points, int3* tris, int2* cons, int numPoints, int numTriangles, int numCons)
{
	int* codes;
	int* labels;
	int* codesSorted, * labelsSorted;
	Point2d* pointsSorted;
	utils::malloc(codes, numPoints);
	utils::malloc(labels, numPoints);
	utils::malloc(codesSorted, numPoints);
	utils::malloc(labelsSorted, numPoints);
	utils::malloc(pointsSorted, numPoints);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::kernel(reorderPoint_s1, numPoints, 128, points, labels, codes, logPointRes, numPoints);
	//utils::kernel(reorderFace_s1, numTriangles, 128, points, tris, labels, codes, logPointRes, numTriangles);

	utils::sort(cubTemp, cubBytes, codes, codesSorted, labels, labelsSorted, numPoints);

	utils::kernel(reorderPoint_s2, numPoints, 128, points, pointsSorted, labels, labelsSorted, numPoints);

	if (tris != nullptr) {
		utils::kernel(reorderPoint_s3, numTriangles, 128, tris, labels, numTriangles);
	}
	if (cons != nullptr)
	{
		utils::kernel(reorderPoint_s4, numCons, 128, cons, labels, numCons);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "reorder points" << " time: " << elapsedTime << " ms\n";

	utils::release(codes, numPoints);
	utils::release(labels, numPoints);
	utils::release(codesSorted, numPoints);
	utils::release(labelsSorted, numPoints);

	utils::release(points, numPoints);

	points = pointsSorted;
	return elapsedTime;
}

__global__ void clearNeedTest(int* needTest, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	needTest[tid] = 1;
}

__device__ int getId(const int3& t, const int2& e)
{
	if (t.x != e.x && t.x != e.y) return 0;
	if (t.y != e.x && t.y != e.y) return 1;
	return 2;
}
__device__ int getId(const int3& f1, const int3& f2)
{
	if (!contains(f2, f1.x)) return 0;
	if (!contains(f2, f1.y)) return 1;
	return 2;
}

__global__ void getSideV(int2* edges, int4* edge2edge, int2* edgeSide, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int e2ey = edge2edge[tid].y;
	int2 ret = { -1, -1 };
	if (e2ey != -1) {
		ret.x = getInt2(edges + e2ey / 2, e2ey % 2);
	}
	int e2ew = edge2edge[tid].w;
	if (e2ew != -1) {
		ret.y = getInt2(edges + e2ew / 2, e2ew % 2);
	}
	edgeSide[tid] = ret;
}

__global__ void initNeedFlip(Point2d* points, int3* face2Vertex, int2* edge2face, int* needFlip, int* atomicFace, int* d_mark, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 e2f = edge2face[tid];
	int p = -1;
	if (e2f.x != -1 && e2f.y != -1)
	{
		int3 f1 = face2Vertex[e2f.x];
		int3 f2 = face2Vertex[e2f.y];
		int ov;
		if (!contains(f1, f2.x))
		{
			ov = f2.x;
		}
		else if (!contains(f1, f2.y))
		{
			ov = f2.y;
		}
		else {
			ov = f2.z;
		}
		p = inCircle(points[f1.x], points[f1.y], points[f1.z], points[ov]);
		if (p > 0)
		{
			d_mark[0] = 1;
			//atomicFace[e2f.x * 3 + getId(f1, f2)] = tid;
			//atomicFace[e2f.y * 3 + getId(f2, f1)] = tid;
			atomicMax(atomicFace + e2f.x, tid);
			atomicMax(atomicFace + e2f.y, tid);
		}
	}
	needFlip[tid] = p > 0 ? 1 : 0;
}

__global__ void initNeedFlip_n(Point2d* points, int2* edgeV, int2* edgeSideV, int2* edge2face, int* needFlip, int* atomicFace, int* d_mark, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 e2f = edge2face[tid];
	int p = -1;
	if (e2f.x != -1 && e2f.y != -1)
	{
		int2 ev = edgeV[tid];
		int2 es = edgeSideV[tid];

		p = inCircle(points[es.x], points[ev.x], points[ev.y], points[es.y]);
		if (p > 0)
		{
			d_mark[0] = 1;
			atomicMax(atomicFace + e2f.x, tid);
			atomicMax(atomicFace + e2f.y, tid);
		}
	}
	needFlip[tid] = p > 0 ? 1 : 0;
}

__global__ void initNeedFlip_new(Point2d* points, Point2d2* edges, int2* edge2face, int4* edge2edge, int* needTest, int* needFlip, int* atomicFace, int* d_mark, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	if (needTest[tid] != -1) return;

	int2 e2f = edge2face[tid];
	int p = -1;
	if (e2f.x != -1 && e2f.y != -1)
	{
		int e2ey = edge2edge[tid].y;
		int e2ew = edge2edge[tid].w;
		Point2d2 ey, ew;
		reinterpret_cast<int4*>(&ey)[0] = reinterpret_cast<int4*>(edges)[e2ey / 2];
		reinterpret_cast<int4*>(&ew)[0] = reinterpret_cast<int4*>(edges)[e2ew / 2];
		//Point2d2 ew = edges[e2ew / 2];
		Point2d p1 = getPoint2d2(&ey, e2ey % 2);
		Point2d evx = getPoint2d2(&ey, 1 - (e2ey % 2));
		Point2d p2 = getPoint2d2(&ew, e2ew % 2);
		Point2d evy = getPoint2d2(&ew, 1 - (e2ew % 2));
		p = inCircle(p1, evx, evy, p2);
		if (p > 0)
		{
			d_mark[0] = 1;
			atomicMax(atomicFace + e2f.x, tid);
			atomicMax(atomicFace + e2f.y, tid);
		}
	}
	needFlip[tid] = p > 0 ? 1 : 0;
	needTest[tid] = 0;
}

__global__ void initNeedFlipFE_new(Point2d* points, int2* edges, int2* edge2face, int3* face2edge, int3* face2vert, int* needTest, int* needFlip, int* noFlips, int* atomicFace, int* d_mark, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	if (needTest[tid] != -1) return;
	if (noFlips[tid] == 1) {
		needFlip[tid] = 0; return;
	}
	int2 e2f = edge2face[tid];
	int p = -1;
	if (e2f.x != -1 && e2f.y != -1)
	{
		int3 f2v = face2vert[e2f.x / 3];
		//int3 f2v2 = face2vert[e2f.y];
		//int other;
		//if (!contains(f2v, f2v2.x)) other = f2v2.x;
		//else if (!contains(f2v, f2v2.y)) other = f2v2.y;
		//else other = f2v2.z;
		int other = getInt3(face2vert + e2f.y / 3, e2f.y % 3);
		//printf("%d: %d %d %d %d %d  %d\n", tid, e2f.x, e2f.y, f2v.x, f2v.y, f2v.z, other);
		p = inCircle(points[f2v.x], points[f2v.y], points[f2v.z], points[other]);
		if (p > 0)
		{
			d_mark[0] = 1;
			atomicMax(atomicFace + e2f.x / 3, tid);
			atomicMax(atomicFace + e2f.y / 3, tid);
		}
	}
	needFlip[tid] = p > 0 ? 1 : 0;
	needTest[tid] = 0;
}

__global__ void flipEdge_healAndtest(Point2d* points, int3* face2Vertex, int2* edges, int2* edge2face, int* futureFace, int* needFlip, int* d_mark, int* atomicFace, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 ev = edges[tid];
	int2 e2f = edge2face[tid];
	int3 f1, f2;
	f1.x = -1; f2.x = -1;
	int nt = 0;
	if (e2f.x != -1)
	{
		int ff = futureFace[e2f.x];
		if (ff != -1) {
			if (ff != e2f.y) {
				nt = 1;
			}
			f1 = face2Vertex[e2f.x];
			//if (!contains(f1, ev.x) || !contains(f1, ev.y))
			//{
			//	e2f.x = ff;
			//}
			if (!contains(f1, ev.x))
			{
				e2f.x = ff;
				f1.x = ev.x;
				int t = f1.y; f1.y = f1.z; f1.z = t;
				edge2face[tid].x = e2f.x;
			}
			else if (!contains(f1, ev.y))
			{
				e2f.x = ff;
				f1.x = ev.y;
				int t = f1.y; f1.y = f1.z; f1.z = t;
				edge2face[tid].x = e2f.x;
			}
		}
	}

	if (e2f.y != -1)
	{
		int ff = futureFace[e2f.y];
		if (ff != -1) {
			if (ff != e2f.x) {
				nt = 1;
			}
			f2 = face2Vertex[e2f.y];
			//if (!contains(f2, ev.x) || !contains(f2, ev.y))
			//{
			//	e2f.y = ff;
			//}
			if (!contains(f2, ev.x))
			{
				e2f.y = ff;
				f2.x = ev.x;
				int t = f2.y; f2.y = f2.z; f2.z = t;
				edge2face[tid].y = e2f.y;
			}
			else if (!contains(f2, ev.y))
			{
				e2f.y = ff;
				f2.x = ev.y;
				int t = f2.y; f2.y = f2.z; f2.z = t;
				edge2face[tid].y = e2f.y;
			}
		}
	}

	//edge2face[tid] = e2f;
	if (nt > 0) {
		int p = -1;
		if (e2f.x != -1 && e2f.y != -1)
		{
			if (f1.x == -1)
				f1 = face2Vertex[e2f.x];
			if (f2.x == -1)
				f2 = face2Vertex[e2f.y];
			int ov;
			if (!contains(f1, f2.x))
			{
				ov = f2.x;
			}
			else if (!contains(f1, f2.y))
			{
				ov = f2.y;
			}
			else {
				ov = f2.z;
			}
			p = inCircle(points[f1.x], points[f1.y], points[f1.z], points[ov]);
		}

		needFlip[tid] = p > 0 ? 1 : 0;
	}

	if (needFlip[tid] == 1)
	{
		d_mark[0] = 1;
		//atomicFace[e2f.x * 3 + getId(f1, f2)] = tid;
		//atomicFace[e2f.y * 3 + getId(f2, f1)] = tid;
		atomicMax(atomicFace + e2f.x, tid);
		atomicMax(atomicFace + e2f.y, tid);
	}
}

__global__ void flipEdge_healAndtest_n(Point2d* points, int2* edges, int2* edgesSideV, int2* edge2face, int* futurePoint, int* needFlip, int* d_mark, int* atomicFace, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 ev = edges[tid];
	int2 es = edgesSideV[tid];
	int2 e2f = edge2face[tid];

	int nt = 0;
	if (e2f.x != -1)
	{
		int eid = futurePoint[e2f.x];
		if (eid != -1 && eid != tid) {
			nt = 1;
			int2 fev = edges[eid];
			if (fev.y == ev.y)
			{
				es.x = fev.x;
				e2f.x = edge2face[eid].y;
				edge2face[tid].x = e2f.x;
			}
			else if (fev.x == ev.y)
			{
				es.x = fev.y;
				e2f.x = edge2face[eid].x;
				edge2face[tid].x = e2f.x;
			}
			else if (fev.y == ev.x) {
				es.x = fev.x;
			}
			else if (fev.x == ev.x)
			{
				es.x = fev.y;
			}
			edgesSideV[tid].x = es.x;
		}
	}

	if (e2f.y != -1)
	{
		int eid = futurePoint[e2f.y];
		if (eid != -1 && eid != tid) {
			nt = 1;
			int2 fev = edges[eid];
			if (fev.y == ev.x)
			{
				es.y = fev.x;
				e2f.y = edge2face[eid].y;
				edge2face[tid].y = e2f.y;
			}
			else if (fev.x == ev.x)
			{
				es.y = fev.y;
				e2f.y = edge2face[eid].x;
				edge2face[tid].y = e2f.y;
			}
			else if (fev.y == ev.y) {
				es.y = fev.x;
			}
			else if (fev.x == ev.y)
			{
				es.y = fev.y;
			}
			edgesSideV[tid].y = es.y;
		}
	}

	if (nt > 0) {
		int p = -1;
		if (e2f.x != -1 && e2f.y != -1)
		{
			p = inCircle(points[es.x], points[ev.x], points[ev.y], points[es.y]);
		}

		needFlip[tid] = p > 0 ? 1 : 0;
	}

	if (needFlip[tid] == 1)
	{
		d_mark[0] = 1;
		//atomicFace[e2f.x * 3 + getId(f1, f2)] = tid;
		//atomicFace[e2f.y * 3 + getId(f2, f1)] = tid;
		atomicMax(atomicFace + e2f.x, tid);
		atomicMax(atomicFace + e2f.y, tid);
	}
}

__global__ void flipEdge_testAndflip(int3* face2Vertex, int2* edges, int2* edge2face, int* needFlip, int* atomicFace, int* futureFace, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int can = 0;
	if (needFlip[tid] == 1) {
		int2 e2f = edge2face[tid];
		//bool flag = true;
		//for (int i = 0; i < 3; i++)
		//{
		//	if (atomicFace[e2f.x * 3 + i] > tid) {
		//		flag = false; break;
		//	}
		//	if (atomicFace[e2f.y * 3 + i] > tid) {
		//		flag = false; break;
		//	}
		//}
		//if (flag)
		if (atomicFace[e2f.x] == tid && atomicFace[e2f.y] == tid)
		{
			//atomicFace[e2f.x] = -1;
			//atomicFace[e2f.y] = -1;
			needFlip[tid] = 0;
			futureFace[e2f.x] = e2f.y;
			futureFace[e2f.y] = e2f.x;

			int p1, p2, p3, p4;

			int2 e2v = edges[tid];
			int3 fv1 = face2Vertex[e2f.x];
			int3 fv2 = face2Vertex[e2f.y];
			p1 = fv1.x ^ fv1.y ^ fv1.z ^ e2v.x ^ e2v.y;
			p3 = fv2.x ^ fv2.y ^ fv2.z ^ e2v.x ^ e2v.y;
			if (p1 == fv1.x)
			{
				p2 = fv1.y; p4 = fv1.z;
			}
			else if (p1 == fv1.y) {
				p2 = fv1.z; p4 = fv1.x;
			}
			else {
				p2 = fv1.x; p4 = fv1.y;
			}

			edges[tid] = make_int2(p1, p3);
			int3 f2v;

			f2v.x = p2;
			f2v.y = p3;
			f2v.z = p1;
			face2Vertex[e2f.x] = f2v;

			f2v.x = p4;
			f2v.y = p1;
			f2v.z = p3;
			face2Vertex[e2f.y] = f2v;
		}
	}
}

__global__ void flipEdge_testAndflip_n(int2* edgeV, int2* edgeSideV, int2* edge2face, int* needFlip, int* atomicFace, int* futureFace, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int can = 0;
	if (needFlip[tid] == 1) {
		int2 e2f = edge2face[tid];
		if (atomicFace[e2f.x] == tid && atomicFace[e2f.y] == tid)
		{
			needFlip[tid] = 0;
			futureFace[e2f.x] = tid;
			futureFace[e2f.y] = tid;

			int2 ev = edgeV[tid];
			int2 es = edgeSideV[tid];

			edgeV[tid] = make_int2(es.y, es.x);
			edgeSideV[tid] = make_int2(ev.x, ev.y);
		}
	}
}

__global__ void flipEdge_solveConflict(int2* edge2face, int* needFlip, int* atomicFace, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	if (needFlip[tid] == 1)
	{
		int2 e2f = edge2face[tid];
		if (atomicFace[e2f.x] != tid || atomicFace[e2f.y] != tid)
		{
			needFlip[tid] = 0;
		}
		else {
			atomicFace[e2f.x] = -2;
			atomicFace[e2f.y] = -2;
		}
	}
}

__global__ void rebuildEdgeInfo(int3* tris, int2* edgeV, int2* edgeSideV, int2* edge2face, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 e2f = edge2face[tid];
	int2 ev = edgeV[tid];
	int2 es = make_int2(-1, -1);
	if (e2f.x != -1)
	{
		int3 t = tris[e2f.x];
		if (t.x != ev.x && t.x != ev.y)
		{
			es.x = t.x;
			if (ev.x != t.y) {
				int t = ev.x; ev.x = ev.y; ev.y = t;
			}
		}
		else if (t.y != ev.x && t.y != ev.y)
		{
			es.x = t.y;
			if (ev.x != t.z) {
				int t = ev.x; ev.x = ev.y; ev.y = t;
			}
		}
		else {
			es.x = t.z;
			if (ev.x != t.x) {
				int t = ev.x; ev.x = ev.y; ev.y = t;
			}
		}
	}

	if (e2f.y != -1)
	{
		int3 t = tris[e2f.y];
		if (t.x != ev.x && t.x != ev.y)
		{
			es.y = t.x;
			if (ev.y != t.y) {
				int t = ev.x; ev.x = ev.y; ev.y = t;
			}
		}
		else if (t.y != ev.x && t.y != ev.y)
		{
			es.y = t.y;
			if (ev.y != t.z) {
				int t = ev.x; ev.x = ev.y; ev.y = t;
			}
		}
		else {
			es.y = t.z;
			if (ev.y != t.x) {
				int t = ev.x; ev.x = ev.y; ev.y = t;
			}
		}
	}
	edgeSideV[tid] = es;
	edgeV[tid] = ev;

	//printf("tid: %d  %d %d  %d %d  %d %d\n", e2f.x, e2f.y, ev.x, ev.y, es.x, es.y);
}

__global__ void rebuildFaceInfo(int3* tris, int2* edgeV, int2* edgeSideV, int2* edge2face, int* atomicFace, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 e2f = edge2face[tid];
	int2 ev = edgeV[tid];
	int2 es = edgeSideV[tid];

	if (e2f.x != -1) {
		int ad = atomicAdd(atomicFace + e2f.x, 1);
		if (ad == 0)
		{
			int3 fx = make_int3(es.x, ev.x, ev.y);
			tris[e2f.x] = fx;
		}
	}

	if (e2f.y != -1)
	{
		int ad = atomicAdd(atomicFace + e2f.y, 1);
		if (ad == 0)
		{
			int3 fy = make_int3(es.y, ev.y, ev.x);
			tris[e2f.y] = fy;
		}
	}
}

__global__ void rebuildFaceAdjInfo(int3* tris, int3* adjTris, int2* edgeSideV, int2* edge2face, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 e2f = edge2face[tid];
	int2 es = edgeSideV[tid];

	if (e2f.x != -1) {
		int3 t = tris[e2f.x];
		if (t.x == es.x) {
			adjTris[e2f.x].x = e2f.y;
		}
		else if (t.y == es.x) {
			adjTris[e2f.x].y = e2f.y;
		}
		else {
			adjTris[e2f.x].z = e2f.y;
		}
	}

	if (e2f.y != -1) {
		int3 t = tris[e2f.y];
		if (t.x == es.y) {
			adjTris[e2f.y].x = e2f.x;
		}
		else if (t.y == es.y) {
			adjTris[e2f.y].y = e2f.x;
		}
		else {
			adjTris[e2f.y].z = e2f.x;
		}
	}
}

//ete is OLD one!
__device__ void flipEdgeNeighbor(int2* edge2face, Point2d2* edges, int4* edge2edge, int2 e2f, int4 e2e, int eid)
{
	getInt2(edge2face + e2e.x / 2, e2e.x % 2) = e2f.y;
	getInt2(edge2face + e2e.z / 2, e2e.z % 2) = e2f.x;

	getInt4AsInt2(edge2edge + e2e.x / 2, e2e.x % 2) = { eid * 2 + 1, e2e.w };
	getInt4AsInt2(edge2edge + e2e.z / 2, e2e.z % 2) = { eid * 2, e2e.y };

	getInt4AsInt2(edge2edge + e2e.y / 2, e2e.y % 2) = { e2e.z, eid * 2 };
	getInt4AsInt2(edge2edge + e2e.w / 2, e2e.w % 2) = { e2e.x, eid * 2 + 1 };
}

__device__ void flipEdgeNeighbor(int2* edge2face, int2* edges, int2* edgeSide, int4* edge2edge, int np1, int np2, int2 e2f, int4 e2e, int eid)
{
	getInt2(edge2face + e2e.x / 2, e2e.x % 2) = e2f.y;
	getInt2(edge2face + e2e.z / 2, e2e.z % 2) = e2f.x;

	getInt2(edgeSide + e2e.x / 2, e2e.x % 2) = np2;
	getInt2(edgeSide + e2e.y / 2, e2e.y % 2) = np2;
	getInt2(edgeSide + e2e.z / 2, e2e.z % 2) = np1;
	getInt2(edgeSide + e2e.w / 2, e2e.w % 2) = np1;
	getInt4AsInt2(edge2edge + e2e.x / 2, e2e.x % 2) = { eid * 2 + 1, e2e.w };
	getInt4AsInt2(edge2edge + e2e.z / 2, e2e.z % 2) = { eid * 2, e2e.y };

	getInt4AsInt2(edge2edge + e2e.y / 2, e2e.y % 2) = { e2e.z, eid * 2 };
	getInt4AsInt2(edge2edge + e2e.w / 2, e2e.w % 2) = { e2e.x, eid * 2 + 1 };
}

__device__ void flipEdgeFE(int2* edge2face, int3* face2edge, int3* face2vert, int4& exe, int4& exp, int2& e2f, int eid)
{
	//printf("flip %d %d %d %d %d\n", eid, exe.x / 2, exe.z / 2, e2f.x, e2f.y);
	getInt2(edge2face + exe.x / 2, exe.x % 2) = e2f.y;
	getInt2(edge2face + exe.z / 2, exe.z % 2) = e2f.x;

	int xbase = e2f.x % 3;
	int ybase = e2f.y % 3;
	int3 temp = { exe.z, eid * 2, exe.y };
	//face2edge[e2f.x / 3] = { eid * 2, exe.y, exe.z };
	face2edge[e2f.x / 3] = { getInt3(&temp, (3 - xbase) % 3), getInt3(&temp, (4 - xbase) % 3),  getInt3(&temp, (5 - xbase) % 3) };
	
	temp = {exp.x, exp.y, exp.z};
	face2vert[e2f.x / 3] = { getInt3(&temp, (3 - xbase)%3), getInt3(&temp, (4 - xbase) % 3),  getInt3(&temp, (5 - xbase) % 3) };
	//face2vert[e2f.x / 3] = { exp.y, exp.z, exp.x };
	
	temp = { exe.x, eid * 2 + 1, exe.w };
	//face2edge[e2f.y / 3] = { eid * 2 + 1, exe.w, exe.x };
	face2edge[e2f.y / 3] = { getInt3(&temp, (3 - ybase) % 3), getInt3(&temp, (4 - ybase) % 3),  getInt3(&temp, (5 - ybase) % 3) };
	
	temp = { exp.z, exp.w, exp.x };
	face2vert[e2f.y / 3] = { getInt3(&temp, (3 - ybase)%3), getInt3(&temp, (4 - ybase) % 3),  getInt3(&temp, (5 - ybase) % 3) };
	//face2vert[e2f.y / 3] = { exp.w, exp.x, exp.z };
	
	exe = { exe.y, exe.z, exe.w, exe.x };
	exp = { exp.y, exp.z, exp.w, exp.x };
	e2f.x = e2f.x / 3 * 3 + ((e2f.x % 3) + 1) % 3;
	e2f.y = e2f.y / 3 * 3 + ((e2f.y % 3) + 1) % 3;
	edge2face[eid] = e2f;
}
__device__ void flipEdgeSelf(int2* edges, int2* edgeSide, int4* edge2edge, int4& e2e, int x, int y, int p1, int p2, int eid)
{
	edgeSide[eid] = {x, y};
	edges[eid] = { p2, p1 };
	e2e = edge2edge[eid] = { e2e.y, e2e.z, e2e.w, e2e.x };
}

__device__ void flipEdgeSelf(Point2d2* edges, int4* edge2edge, int4& e2e, Point2d p1, Point2d p2, int eid)
{
	edges[eid] = { p2, p1 };
	e2e = edge2edge[eid] = {e2e.y, e2e.z, e2e.w, e2e.x};
}
__device__ Point2d getOtherPoint(Point2d2* edges, int4* edge2edge, int eid)
{
	int ex = getInt4(edge2edge + eid / 2, 2 - (eid % 2));
	if (ex != -1)
	{
		return getPoint2d2(edges + ex / 2, !((ex % 2) ^ (eid % 2)));
	}
	else return {-1, -1};
}
/*
__global__ void testDelaunay(Point2d* points, int2* edges, int4* edge2edge, int* needFlip, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	int4 e2e = edge2edge[tid];
	int2 ev = edges[tid];

	if (e2e.y != -1 && e2e.w != -1)
	{
		const int p1 = getInt2(edges + e2e.y / 2, e2e.y % 2);
		const int p2 = getInt2(edges + e2e.w / 2, e2e.w % 2);

		Point2d P1 = points[p1];
		Point2d Px = points[ev.x];
		Point2d Py = points[ev.y];
		Point2d P2 = points[p2];
		if (inCircle(P1, Px, Py, P2) > 0)
		{
			needFlip[tid] = 1;
		}
		else {
			needFlip[tid] = 0;
		}
		int op;
		if ((op = getOtherPoint(edges, edge2edge, e2e.x)) != -1)
		{
			if (inCircle(points[op], P1, Py, Px) > 0)
			{
				needFlip[e2e.x / 2] = 1;
			}
			else {
				needFlip[e2e.x / 2] = 0;
			}
		}
		if ((op = getOtherPoint(edges, edge2edge, e2e.y)) != -1)
		{
			if (inCircle(points[op], Px, P1, Py) > 0)
			{
				needFlip[e2e.y / 2] = 1;
			}
			else {
				needFlip[e2e.y / 2] = 0;
			}
		}
		if ((op = getOtherPoint(edges, edge2edge, e2e.z)) != -1)
		{
			if (inCircle(points[op], P2, Py, Px) > 0)
			{
				needFlip[e2e.z / 2] = 1;
			}
			else {
				needFlip[e2e.z / 2] = 0;
			}
		}
		if ((op = getOtherPoint(edges, edge2edge, e2e.w)) != -1)
		{
			if (inCircle(points[op], Py, P2, Px) > 0)
			{
				needFlip[e2e.w / 2] = 1;
			}
			else {
				needFlip[e2e.w / 2] = 0;
			}
		}
	}
	else {
		needFlip[tid] = 0;
	}
	
}
*/
__global__ void confirmFlip(int2* edge2face, int* atomicFace, int* needFlip, int* canFlip, int* flipped, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	if (needFlip[tid] == 1) {
		int2 e2f = edge2face[tid];
		if (e2f.x != -1 && e2f.y != -1 && atomicFace[e2f.x / 3] == tid && atomicFace[e2f.y / 3] == tid)
		{
			canFlip[tid] = 1;
			flipped[e2f.x / 3] = tid;
			flipped[e2f.y / 3] = tid;
			atomicFace[e2f.x / 3] = -2;
			atomicFace[e2f.y / 3] = -2;
		}
		else {
			canFlip[tid] = 0;
		}
	}
	else {
		canFlip[tid] = 0;
	}
}

#define QUEUE_SIZE 8
/*
__global__ void multiFlipWithStack(Point2d* points, int2* edge2face, int2* edges, int4* edge2edge, int* atomicFace, int* canFlip, int* flipId, int* needTest, int* queue, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= canFlip[numE]) return;
	int stackBase = tid * QUEUE_SIZE;
	int ocid = tid;
	tid = flipId[tid];
	int stackTop;
	
	{
		int4 e2e = edge2edge[tid];
		int2 e2f = edge2face[tid];

		flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, tid);
		int p1 = getInt2(edges + e2e.y / 2, e2e.y % 2);
		int p2 = getInt2(edges + e2e.w / 2, e2e.w % 2);
		flipEdgeSelf(edges, edge2edge, e2e, p1, p2, tid);
		needTest[tid] = 0;
		needTest[e2e.x / 2] = -1;
		needTest[e2e.y / 2] = -1;
		needTest[e2e.z / 2] = -1;
		needTest[e2e.w / 2] = -1;
	}
	
	int now = tid * 2;
	stackTop = 0;
	queue[stackBase + stackTop] = tid * 2 + 1;
	stackTop = 1;
	int c = 1;
	while (true)
	{
		//if (tid % 1000000 == 0) printf("%d %d %d\n", tid, queueL, queueR);
		int4 e2e = edge2edge[now / 2];
		Point2d P[3];
		int2 ev = edges[now / 2];
		if (now % 2)
		{
			P[0] = points[ev.y];
			P[1] = points[ev.x];
			P[2] = points[getInt2(edges + e2e.w / 2, e2e.w % 2)];
		}
		else
		{
			P[0] = points[ev.x];
			P[1] = points[ev.y];
			P[2] = points[getInt2(edges + e2e.y / 2, e2e.y % 2)];
		}
	
		bool find = false;
#pragma unroll
		for (int i = 0; i < 2; i++)
		{
			int neid = getInt4(&e2e, (now % 2) * 2 + i);
			int fid = getInt2(edge2face + neid / 2, 1 - (neid % 2));
			if (fid != -1)
			{
				int nx = getOtherPoint(edges, edge2edge, neid);
				if (inCircle(points[nx], P[2 - i * 2], P[1 + i], P[0 + i]) > 0)
				{
					//int lock = atomicExch(atomicFace + fid, -2);
					int lock = atomicCAS(atomicFace + fid, -1, tid);
					//if (lock != -2)
					if (lock == -1 || lock == tid)
					{
						int feid = neid / 2;
						int2 np;
						getInt2(&np, neid % 2) = getInt2(&ev, (now % 2) ^ i);
						getInt2(&np, 1 - (neid % 2)) = nx;
						e2e = edge2edge[feid];
						flipEdgeNeighbor(edge2face, edges, edge2edge, edge2face[feid], e2e, feid);
						flipEdgeSelf(edges, edge2edge, e2e, np.x, np.y, feid);
						needTest[feid] = 0;
						needTest[e2e.x / 2] = -1;
						needTest[e2e.y / 2] = -1;
						needTest[e2e.z / 2] = -1;
						needTest[e2e.w / 2] = -1;
						find = true;
						now = feid * 2;
						if (stackTop != QUEUE_SIZE)
						{
							queue[stackBase + stackTop] = now + 1;
							stackTop++;
						}
						c++;
						break;
					}
				}
			}
		}
	
		if (find == false)
		{
			if (stackTop == 0) break;
			now = queue[stackBase + stackTop - 1];
			stackTop--;
		}
	}
	//if (c > 10) printf("%d\n", c);
	canFlip[ocid] = c;
}
*/
/*
__global__ void multiFlipWithStack2(Point2d* points, int2* edge2face, int2* edges, int4* edge2edge, int* atomicFace, int* canFlip, int* flipId, int* needTest, int* queue, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= canFlip[numE]) return;
	int ocid = tid;
	tid = flipId[tid];
	int4 e2e = edge2edge[tid];
	{
		
		int2 e2f = edge2face[tid];

		flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, tid);
		int p1 = getInt2(edges + e2e.y / 2, e2e.y % 2);
		int p2 = getInt2(edges + e2e.w / 2, e2e.w % 2);
		flipEdgeSelf(edges, edge2edge, e2e, p1, p2, tid);
		needTest[tid] = 0;
		needTest[e2e.x / 2] = -1;
		needTest[e2e.y / 2] = -1;
		needTest[e2e.z / 2] = -1;
		needTest[e2e.w / 2] = -1;
	}

	int now = tid;
	int c = 1;
	bool find = true;
	Point2d P[3];
	int2 ev = edges[now];
	//int p1 = getInt2(edges + e2e.y / 2, e2e.y % 2);
	//int p2 = getInt2(edges + e2e.w / 2, e2e.w % 2);
	//P[0] = points[p1];
	//P[1] = points[ev.x];
	//P[2] = points[p2];
	//P[3] = points[ev.y];
	while (find)
	{
		//ev = edges[now];
		find = false;
#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			P[i / 2] = points[ev.x];
			P[1 - i / 2] = points[ev.y];
			{
				int temp = getInt4(&e2e, (i / 2) * 2 + 1);
				P[2] = points[getInt2(edges + temp / 2, temp % 2)];
			}
			int neid = getInt4(&e2e, i);
			int fid = getInt2(edge2face + neid / 2, 1 - (neid % 2));
			
			if (fid != -1 && atomicFace[fid] != -2)
			{
				int nx = getOtherPoint(edges, edge2edge, neid);
				if (inCircle(points[nx], P[2 - (i%2) * 2], P[1 + i % 2], P[i % 2]) > 0)
				//if (inCircle(P[i/2], P[1 + i/2], P[3], NP) > 0)
				{
					//int lock = atomicExch(atomicFace + fid, -2);
					int lock = atomicCAS(atomicFace + fid, -1, tid);
					//if (lock != -2)
					if (lock == -1 || lock == tid)
					{
						int feid = neid / 2;
						int2 np;
						getInt2(&np, neid % 2) = getInt2(&ev, (i / 2) ^ (i % 2));
						getInt2(&np, 1 - (neid % 2)) = nx;
						e2e = edge2edge[feid];
						flipEdgeNeighbor(edge2face, edges, edge2edge, edge2face[feid], e2e, feid);
						flipEdgeSelf(edges, edge2edge, e2e, np.x, np.y, feid);
						ev = { np.y, np.x };
						//Point2d tP[4];
						//tP[0] = P[0]; tP[1] = P[1]; tP[2] = P[2]; tP[3] = P[3];
						//P[0] = tP[(i + 3 + neid % 2) % 4];
						//P[2] = tP[(i + 4 - neid % 2) % 4];
						//P[(neid % 2) * 2 + 1] = NP;
						//P[3 - (neid % 2) * 2] = tP[1 + 2 * ((i & 1) ^ (i >> 1))];
						needTest[feid] = 0;
						needTest[e2e.x / 2] = -1;
						needTest[e2e.y / 2] = -1;
						needTest[e2e.z / 2] = -1;
						needTest[e2e.w / 2] = -1;
						find = true;
						now = feid;
						c++;
						break;
					}
				}
			}
		}
	}

	e2e = edge2edge[tid];
	now = tid;
	find = true;
	ev = edges[now];
	//p1 = getInt2(edges + e2e.y / 2, e2e.y % 2);
	//p2 = getInt2(edges + e2e.w / 2, e2e.w % 2);
	//P[0] = points[p1];
	//P[1] = points[ev.x];
	//P[2] = points[p2];
	//P[3] = points[ev.y];
	while (find)
	{
		find = false;
#pragma unroll
		for (int i = 3; i >= 0; i--)
		{
			P[i / 2] = points[ev.x];
			P[1 - i / 2] = points[ev.y];
			{
				int temp = getInt4(&e2e, (i / 2) * 2 + 1);
				P[2] = points[getInt2(edges + temp / 2, temp % 2)];
			}
			int neid = getInt4(&e2e, i);
			int fid = getInt2(edge2face + neid / 2, 1 - (neid % 2));

			if (fid != -1 && atomicFace[fid] != -2)
			{
				int nx = getOtherPoint(edges, edge2edge, neid);
				if (inCircle(points[nx], P[2 - (i % 2) * 2], P[1 + i % 2], P[i % 2]) > 0)
				//Point2d NP = points[nx];
				//if (inCircle(P[i / 2], P[1 + i / 2], P[3], NP) > 0)
				{
					//int lock = atomicExch(atomicFace + fid, -2);
					int lock = atomicCAS(atomicFace + fid, -1, tid);
					//if (lock != -2)
					if (lock == -1 || lock == tid)
					{
						int feid = neid / 2;
						int2 np;
						getInt2(&np, neid % 2) = getInt2(&ev, (i / 2) ^ (i % 2));
						getInt2(&np, 1 - (neid % 2)) = nx;
						e2e = edge2edge[feid];
						flipEdgeNeighbor(edge2face, edges, edge2edge, edge2face[feid], e2e, feid);
						flipEdgeSelf(edges, edge2edge, e2e, np.x, np.y, feid);
						ev = { np.y, np.x };
						//Point2d tP[4];
						//tP[0] = P[0]; tP[1] = P[1]; tP[2] = P[2]; tP[3] = P[3];
						//P[0] = tP[(i + 3 + neid % 2) % 4];
						//P[2] = tP[(i + 4 - neid % 2) % 4];
						//P[(neid % 2) * 2 + 1] = NP;
						//P[3 - (neid % 2) * 2] = tP[1 + 2 * ((i & 1) ^ (i >> 1))];
						needTest[feid] = 0;
						needTest[e2e.x / 2] = -1;
						needTest[e2e.y / 2] = -1;
						needTest[e2e.z / 2] = -1;
						needTest[e2e.w / 2] = -1;
						find = true;
						now = feid;
						c++;
						break;
					}
				}
			}
		}
	}
	//if (c > 10) printf("%d\n", c);
	canFlip[ocid] = c;
}
*/
__global__ void multiFlip(Point2d* points, int2* edge2face, Point2d2* edges, int4* edge2edge, int* atomicFace, int* canFlip, int* flipId, int* needTest, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= canFlip[numE]) return;
	int otid = tid;
	tid = flipId[tid];

	int4 e2e = edge2edge[tid];
	int2 e2f = edge2face[tid];

	flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, tid);
	Point2d Py = getPoint2d2(edges + e2e.y / 2, e2e.y % 2); //p1
	Point2d Px = getPoint2d2(edges + e2e.w / 2, e2e.w % 2); //p2
	//Point2d P1, Px, Py, P2;
	//Py = points[p1]; Px = points[p2];
	//int2 ev = edges[tid];
	//P1 = points[ev.x]; P2 = points[ev.y];
	Point2d P1 = edges[tid].x;
	Point2d P2 = edges[tid].y;
	flipEdgeSelf(edges, edge2edge, e2e, Py, Px, tid);
	//int py = p1, px = p2;
	//p1 = ev.x; p2 = ev.y;
	bool change = true;
	int eid = tid;
	int c = 1;
	while (change) {
		//printf("%d: %d %d %d %d\n", tid, px, py, p1, p2);
		needTest[eid] = 0;
		needTest[e2e.x / 2] = -1;
		needTest[e2e.y / 2] = -1;
		needTest[e2e.z / 2] = -1;
		needTest[e2e.w / 2] = -1;
		change = false;
		int fid = getInt2(edge2face + e2e.x / 2, 1 - (e2e.x % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.x);
			//Point2d Pn = points[nx];
			if (inCircle(Pn, P1, Py, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//printf("flip x\n");
					//FLIP!
					eid = e2e.x / 2;
					Point2d np1, np2;
					if (e2e.x % 2) {
						np1 = Pn;
						np2 = Px;
						//
						//p2 = py;
						//py = nx;
						P2 = Py;
						Py = Pn;
					}
					else {
						np1 = Px;
						np2 = Pn;
						//
						//p2 = p1;
						//p1 = py;
						//py = px;
						//px = nx;
						P2 = P1;
						P1 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.x % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.x % 2)) = fid;
					e2e = edge2edge[eid];
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);
					c++;
					change = true;
	
					continue;
				}
			}
		}
		fid = getInt2(edge2face + e2e.y / 2, 1 - (e2e.y % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.y);
			//Point2d Pn = points[ny];
			if (inCircle(Pn, Px, P1, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.y / 2;
					//printf("flip y\n");
					Point2d np1, np2;
					if (e2e.y % 2) {
						np1 = Pn;
						np2 = Py;
		
						//p2 = p1;
						//p1 = px;
						//px = py;
						//py = ny;
						P2 = P1;
						P1 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = Py;
						np2 = Pn;
		
						//p2 = px;
						//px = ny;
						P2 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.y % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.y % 2)) = fid;
					e2e = edge2edge[eid];
		
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);
		
					change = true;
					c++;
					continue;
				}
			}
		}
		fid = getInt2(edge2face + e2e.z / 2, 1 - (e2e.z % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.z);
			//Point2d Pn = points[nz];
			if (inCircle(Pn, P2, Px, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.z / 2;
					//printf("flip z\n");
					Point2d np1, np2;
					if (e2e.z % 2) {
						np1 = Pn;
						np2 = Py;
	
						//p1 = p2;
						//p2 = px;
						//px = py;
						//py = nz;
						P1 = P2;
						P2 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = Py;
						np2 = Pn;
	
						//p1 = px;
						//px = nz;
						P1 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.z % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.z % 2)) = fid;
					e2e = edge2edge[eid];
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);
	
					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.w / 2, 1 - (e2e.w % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.w);
			//Point2d Pn = points[nw];
			if (inCircle(Pn, Py, P2, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					//printf("flip w\n");
					eid = e2e.w / 2;
					Point2d np1, np2;
					if (e2e.w % 2) {
						np1 = Pn;
						np2 = Px;
	
						//p1 = py;
						//py = nw;
						P1 = Py;
						Py = Pn;
					}
					else {
						np1 = Px;
						np2 = Pn;
	
						//p1 = p2;
						//p2 = py;
						//py = px;
						//px = nw;
	
						P1 = P2;
						P2 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.w % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.w % 2)) = fid;
					e2e = edge2edge[eid];
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);
	
					change = true;
					c++;
					continue;
				}
			}
		}
	}

	change = true;
	e2e = edge2edge[tid];
	e2f = edge2face[tid];
	eid = tid;
	//ev = edges[tid];
	P1 = getPoint2d2(edges + e2e.y / 2, e2e.y % 2);
	P2 = getPoint2d2(edges + e2e.w / 2, e2e.w % 2);
	//P1 = points[p1]; P2 = points[p2];
	//Px = points[ev.x]; Py = points[ev.y];
	Px = edges[tid].x;
	Py = edges[tid].y;
	//py = ev.y, px = ev.x;
	bool first = true;
	while (change)
	{
		//printf("%d: %d %d %d %d\n", tid, px, py, p1, p2);
		if (first) {
			first = false;
		}
		else {
			needTest[eid] = 0;
			needTest[e2e.x / 2] = -1;
			needTest[e2e.y / 2] = -1;
			needTest[e2e.z / 2] = -1;
			needTest[e2e.w / 2] = -1;
		}
		change = false;

		int fid = getInt2(edge2face + e2e.z / 2, 1 - (e2e.z % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.z);
			//Point2d Pn = points[nz];
			if (inCircle(Pn, P2, Px, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.z / 2;
					//printf("flip z\n");
					Point2d np1, np2;
					if (e2e.z % 2) {
						np1 = Pn;
						np2 = Py;

						//p1 = p2;
						//p2 = px;
						//px = py;
						//py = nz;
						P1 = P2;
						P2 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = Py;
						np2 = Pn;

						//p1 = px;
						//px = nz;
						P1 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.z % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.z % 2)) = fid;
					e2e = edge2edge[eid];
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.w / 2, 1 - (e2e.w % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.w);
			//Point2d Pn = points[nw];
			if (inCircle(Pn, Py, P2, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					//printf("flip w\n");
					eid = e2e.w / 2;
					Point2d np1, np2;
					if (e2e.w % 2) {
						np1 = Pn;
						np2 = Px;

						//p1 = py;
						//py = nw;
						P1 = Py;
						Py = Pn;
					}
					else {
						np1 = Px;
						np2 = Pn;

						//p1 = p2;
						//p2 = py;
						//py = px;
						//px = nw;

						P1 = P2;
						P2 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.w % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.w % 2)) = fid;
					e2e = edge2edge[eid];
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.x / 2, 1 - (e2e.x % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.x);
			//Point2d Pn = points[nx];
			if (inCircle(Pn, P1, Py, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//printf("flip x\n");
					//FLIP!
					eid = e2e.x / 2;
					Point2d np1, np2;
					if (e2e.x % 2) {
						np1 = Pn;
						np2 = Px;

						//p2 = py;
						//py = nx;
						P2 = Py;
						Py = Pn;
					}
					else {
						np1 = Px;
						np2 = Pn;

						//p2 = p1;
						//p1 = py;
						//py = px;
						//px = nx;
						P2 = P1;
						P1 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.x % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.x % 2)) = fid;
					e2e = edge2edge[eid];
					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);
					c++;
					change = true;

					continue;
				}
			}
		}
		fid = getInt2(edge2face + e2e.y / 2, 1 - (e2e.y % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			Point2d Pn = getOtherPoint(edges, edge2edge, e2e.y);
			//Point2d Pn = points[ny];
			if (inCircle(Pn, Px, P1, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.y / 2;
					//printf("flip y\n");
					Point2d np1, np2;
					if (e2e.y % 2) {
						np1 = Pn;
						np2 = Py;

						//p2 = p1;
						//p1 = px;
						//px = py;
						//py = ny;
						P2 = P1;
						P1 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = Py;
						np2 = Pn;

						//p2 = px;
						//px = ny;
						P2 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.y % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.y % 2)) = fid;
					e2e = edge2edge[eid];

					flipEdgeNeighbor(edge2face, edges, edge2edge, e2f, e2e, eid);
					flipEdgeSelf(edges, edge2edge, e2e, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}
	}

	canFlip[otid] = c;
}

__device__ int2 getOtherEdges(int3 fe, int eid)
{
	int2 otherEdge;
	if (fe.x / 2 == eid)
	{
		otherEdge.x = fe.y;
		otherEdge.y = fe.z;
	}
	else if (fe.y / 2 == eid)
	{
		otherEdge.x = fe.z;
		otherEdge.y = fe.x;
	}
	else {
		otherEdge.x = fe.x;
		otherEdge.y = fe.y;
	}
	return otherEdge;
}
__global__ void multiFlipFE(Point2d* points, int2* edge2face, int3* face2edge, int3* face2vert, int* atomicFace, int* canFlip, int* flipId, int* needTest, int* noFlips, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= canFlip[numE]) return;
	int otid = tid;
	tid = flipId[tid];

	int4 exe, exp;
	int2 e2f = edge2face[tid];
	//printf("flip %d %d %d\n", tid, e2f.x, e2f.y);
	{
		int3 temp = face2edge[e2f.x / 3];
		if (temp.x/2 == tid)
		{
			exe.x = temp.y;
			exe.y = temp.z;
		}
		else if (temp.y/2 == tid)
		{
			exe.x = temp.z;
			exe.y = temp.x;
		}
		else {
			exe.x = temp.x;
			exe.y = temp.y;
		}
		temp = face2edge[e2f.y / 3];
		if (temp.x/2 == tid)
		{
			exe.z = temp.y;
			exe.w = temp.z;
		}
		else if (temp.y/2 == tid)
		{
			exe.z = temp.z;
			exe.w = temp.x;
		}
		else {
			exe.z = temp.x;
			exe.w = temp.y;
		}
	}
	{
		int3 f1 = face2vert[e2f.x / 3];
		int3 f2 = face2vert[e2f.y / 3];
		if (!contains(f1, f2.x))
		{
			exp.z = f2.x;
			exp.w = f2.y;
			exp.y = f2.z;
		}
		else if (!contains(f1, f2.y))
		{
			exp.z = f2.y;
			exp.w = f2.z;
			exp.y = f2.x;
		}
		else {
			exp.z = f2.z;
			exp.w = f2.x;
			exp.y = f2.y;
		}
		if (!contains(f2, f1.x))
		{
			exp.x = f1.x;
		}
		else if (!contains(f2, f1.y))
		{
			exp.x = f1.y;
		}
		else {
			exp.x = f1.z;
		}
	}
	//printf("before %d %d %d %d %d %d %d\n", tid, exp.x, exp.y, exp.z, exp.w, e2f.x, e2f.y);
	flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, tid);
	//printf("after  %d %d %d %d %d %d %d\n", tid, exp.x, exp.y, exp.z, exp.w, e2f.x, e2f.y);
	Point2d P[4];
	P[0] = points[exp.x];
	P[1] = points[exp.y];
	P[2] = points[exp.z];
	P[3] = points[exp.w];
	bool change = true;
	int eid = tid;
	int c = 1;
	needTest[eid] = 0;
	needTest[exe.x / 2] = -1;
	needTest[exe.y / 2] = -1;
	needTest[exe.z / 2] = -1;
	needTest[exe.w / 2] = -1;

	int count = 0;
	while (change) {
		//printf("%d: %d %d %d %d\n", tid, px, py, p1, p2);
		count++;
		needTest[eid] = 0;
		needTest[exe.x / 2] = -1;
		needTest[exe.y / 2] = -1;
		needTest[exe.z / 2] = -1;
		needTest[exe.w / 2] = -1;
		if (count > 10) break;
		change = false;
		int fid = getInt2(edge2face + exe.x / 2, 1 - (exe.x % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.x / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.x ^ exp.w;
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[0], P[1], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid / 3];
					// otherEdge = getOtherEdges(fe, exe.x / 2);
					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
					                   getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };

					getInt2(&e2f, exe.x % 2) = e2f.x / 3 * 3 + (e2f.x % 3 + 1) % 3;
					getInt2(&e2f, 1 - (exe.x % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.x;
					if (exe.x % 2)
					{
						exp = { pn, exp.x, exp.y, exp.w };
						exe = { otherEdge.x, otherEdge.y, exe.y, eid * 2 };
						P[2] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.y, exp.w, pn, exp.x };
						exe = { exe.y, eid * 2, otherEdge.x, otherEdge.y };
						P[2] = P[0];
						P[0] = P[3];
						P[3] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;
	
					continue;
				}
			}
		}
		
		fid = getInt2(edge2face + exe.y / 2, 1 - (exe.y % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.y / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.x ^ exp.y;
				//Pn = points[pn];
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[0], P[1], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid];
					//int2 otherEdge = getOtherEdges(fe, exe.y / 2);
					//getInt2(&e2f, exe.y % 2) = e2f.x;
					//getInt2(&e2f, 1 - (exe.y % 2)) = fid;

					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
									   getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };
					getInt2(&e2f, exe.y % 2) = e2f.x / 3 * 3 + (e2f.x % 3 + 2) % 3;
					getInt2(&e2f, 1 - (exe.y % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.y;
					if (exe.y % 2)
					{
						exp = { pn, exp.y, exp.w, exp.x };
						exe = { otherEdge.x, otherEdge.y, eid * 2, exe.x };
						P[2] = P[0];
						P[0] = P[1];
						P[1] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.w, exp.x, pn, exp.y };
						exe = { eid * 2, exe.x, otherEdge.x, otherEdge.y };
						P[2] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;

					continue;
				}
			}
		}

		fid = getInt2(edge2face + exe.z / 2, 1 - (exe.z % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.z / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.z ^ exp.y;
				//Pn = points[pn];
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[1], P[2], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid];
					//int2 otherEdge = getOtherEdges(fe, exe.z / 2);
					//getInt2(&e2f, exe.z % 2) = e2f.y;
					//getInt2(&e2f, 1 - (exe.z % 2)) = fid;

					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
									   getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };
					getInt2(&e2f, exe.z % 2) = e2f.y / 3 * 3 + (e2f.y % 3 + 1) % 3;
					getInt2(&e2f, 1 - (exe.z % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.z;
					if (exe.z % 2)
					{
						exp = { pn, exp.z, exp.w, exp.y };
						exe = { otherEdge.x, otherEdge.y, exe.w, eid * 2 + 1 };
						P[0] = P[2];
						P[2] = P[1];
						P[1] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.w, exp.y, pn, exp.z };
						exe = { exe.w, eid * 2 + 1, otherEdge.x, otherEdge.y };
						P[0] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;
		
					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;
		
					continue;
				}
			}
		}

		fid = getInt2(edge2face + exe.w / 2, 1 - (exe.w % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.w / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.z ^ exp.w;
				//Pn = points[pn];
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[1], P[2], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid];
					//int2 otherEdge = getOtherEdges(fe, exe.w / 2);
					//getInt2(&e2f, exe.w % 2) = e2f.y;
					//getInt2(&e2f, 1 - (exe.w % 2)) = fid;
					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
					getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };
					getInt2(&e2f, exe.w % 2) = e2f.y / 3 * 3 + (e2f.y % 3 + 2) % 3;
					getInt2(&e2f, 1 - (exe.w % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.w;
					if (exe.w % 2)
					{
						exp = { pn, exp.w, exp.y, exp.z };
						exe = { otherEdge.x, otherEdge.y, eid * 2 + 1, exe.z};
						P[0] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.y, exp.z, pn, exp.w };
						exe = { eid * 2 + 1, exe.z, otherEdge.x, otherEdge.y };
						P[0] = P[2];
						P[2] = P[3];
						P[3] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;

					continue;
				}
			}
		}
	}
	canFlip[otid] = c;

	e2f = edge2face[tid];
	//printf("flip %d %d %d\n", tid, e2f.x, e2f.y);
	{
		int3 temp = face2edge[e2f.x / 3];
		if (temp.x / 2 == tid)
		{
			exe.x = temp.y;
			exe.y = temp.z;
		}
		else if (temp.y / 2 == tid)
		{
			exe.x = temp.z;
			exe.y = temp.x;
		}
		else {
			exe.x = temp.x;
			exe.y = temp.y;
		}
		temp = face2edge[e2f.y / 3];
		if (temp.x / 2 == tid)
		{
			exe.z = temp.y;
			exe.w = temp.z;
		}
		else if (temp.y / 2 == tid)
		{
			exe.z = temp.z;
			exe.w = temp.x;
		}
		else {
			exe.z = temp.x;
			exe.w = temp.y;
		}
	}
	{
		int3 f1 = face2vert[e2f.x / 3];
		int3 f2 = face2vert[e2f.y / 3];
		if (!contains(f1, f2.x))
		{
			exp.z = f2.x;
			exp.w = f2.y;
			exp.y = f2.z;
		}
		else if (!contains(f1, f2.y))
		{
			exp.z = f2.y;
			exp.w = f2.z;
			exp.y = f2.x;
		}
		else {
			exp.z = f2.z;
			exp.w = f2.x;
			exp.y = f2.y;
		}
		if (!contains(f2, f1.x))
		{
			exp.x = f1.x;
		}
		else if (!contains(f2, f1.y))
		{
			exp.x = f1.y;
		}
		else {
			exp.x = f1.z;
		}
	}
	P[0] = points[exp.x];
	P[1] = points[exp.y];
	P[2] = points[exp.z];
	P[3] = points[exp.w];
	change = true;
	eid = tid;
	bool first = true;
	while (change) {
		count++;
		//printf("%d: %d %d %d %d\n", tid, px, py, p1, p2);
		if (first)
		{
			first = false;
		}
		else {
			needTest[eid] = 0;
			needTest[exe.x / 2] = -1;
			needTest[exe.y / 2] = -1;
			needTest[exe.z / 2] = -1;
			needTest[exe.w / 2] = -1;
		}
		change = false;
		if (count > 10) break; if (count > 10) break;
		int fid = getInt2(edge2face + exe.z / 2, 1 - (exe.z % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.z / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.z ^ exp.y;
				//Pn = points[pn];
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[1], P[2], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid];
					//int2 otherEdge = getOtherEdges(fe, exe.z / 2);
					//getInt2(&e2f, exe.z % 2) = e2f.y;
					//getInt2(&e2f, 1 - (exe.z % 2)) = fid;

					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
									   getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };
					getInt2(&e2f, exe.z % 2) = e2f.y / 3 * 3 + (e2f.y % 3 + 1) % 3;
					getInt2(&e2f, 1 - (exe.z % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.z;
					if (exe.z % 2)
					{
						exp = { pn, exp.z, exp.w, exp.y };
						exe = { otherEdge.x, otherEdge.y, exe.w, eid * 2 + 1 };
						P[0] = P[2];
						P[2] = P[1];
						P[1] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.w, exp.y, pn, exp.z };
						exe = { exe.w, eid * 2 + 1, otherEdge.x, otherEdge.y };
						P[0] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;

					continue;
				}
			}
		}

		fid = getInt2(edge2face + exe.w / 2, 1 - (exe.w % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.w / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.z ^ exp.w;
				//Pn = points[pn];
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[1], P[2], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid];
					//int2 otherEdge = getOtherEdges(fe, exe.w / 2);
					//getInt2(&e2f, exe.w % 2) = e2f.y;
					//getInt2(&e2f, 1 - (exe.w % 2)) = fid;
					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
					getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };
					getInt2(&e2f, exe.w % 2) = e2f.y / 3 * 3 + (e2f.y % 3 + 2) % 3;
					getInt2(&e2f, 1 - (exe.w % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.w;
					if (exe.w % 2)
					{
						exp = { pn, exp.w, exp.y, exp.z };
						exe = { otherEdge.x, otherEdge.y, eid * 2 + 1, exe.z };
						P[0] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.y, exp.z, pn, exp.w };
						exe = { eid * 2 + 1, exe.z, otherEdge.x, otherEdge.y };
						P[0] = P[2];
						P[2] = P[3];
						P[3] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;

					continue;
				}
			}
		}

		fid = getInt2(edge2face + exe.x / 2, 1 - (exe.x % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.x / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.x ^ exp.w;
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[0], P[1], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid / 3];
					// otherEdge = getOtherEdges(fe, exe.x / 2);
					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
									   getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };

					getInt2(&e2f, exe.x % 2) = e2f.x / 3 * 3 + (e2f.x % 3 + 1) % 3;
					getInt2(&e2f, 1 - (exe.x % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.x;
					if (exe.x % 2)
					{
						exp = { pn, exp.x, exp.y, exp.w };
						exe = { otherEdge.x, otherEdge.y, exe.y, eid * 2 };
						P[2] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.y, exp.w, pn, exp.x };
						exe = { exe.y, eid * 2, otherEdge.x, otherEdge.y };
						P[2] = P[0];
						P[0] = P[3];
						P[3] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;

					continue;
				}
			}
		}

		fid = getInt2(edge2face + exe.y / 2, 1 - (exe.y % 2));
		if (fid != -1 && atomicFace[fid / 3] != -2 && noFlips[exe.y / 2] != 1) {
			Point2d Pn;
			int pn;
			{
				//int3 temp = face2vert[fid];
				//pn = temp.x ^ temp.y ^ temp.z ^ exp.x ^ exp.y;
				//Pn = points[pn];
				pn = getInt3(face2vert + fid / 3, fid % 3);
				Pn = points[pn];
			}
			if (inCircle(P[0], P[1], P[3], Pn) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid / 3, -2);
				if (lock != -2)
				{
					//int3 fe = face2edge[fid];
					//int2 otherEdge = getOtherEdges(fe, exe.y / 2);
					//getInt2(&e2f, exe.y % 2) = e2f.x;
					//getInt2(&e2f, 1 - (exe.y % 2)) = fid;

					int2 otherEdge = { getInt3(face2edge + fid / 3, (fid % 3 + 1) % 3),
									   getInt3(face2edge + fid / 3, (fid % 3 + 2) % 3) };
					getInt2(&e2f, exe.y % 2) = e2f.x / 3 * 3 + (e2f.x % 3 + 2) % 3;
					getInt2(&e2f, 1 - (exe.y % 2)) = fid;
					//e2f = edge2face[exe.x / 2];
					int oldE = exe.y;
					if (exe.y % 2)
					{
						exp = { pn, exp.y, exp.w, exp.x };
						exe = { otherEdge.x, otherEdge.y, eid * 2, exe.x };
						P[2] = P[0];
						P[0] = P[1];
						P[1] = P[3];
						P[3] = Pn;
					}
					else {
						exp = { exp.w, exp.x, pn, exp.y };
						exe = { eid * 2, exe.x, otherEdge.x, otherEdge.y };
						P[2] = P[1];
						P[1] = Pn;
					}
					eid = oldE / 2;

					flipEdgeFE(edge2face, face2edge, face2vert, exe, exp, e2f, eid);
					c++;
					change = true;

					continue;
				}
			}
		}	
	}
	//canFlip[otid] = c;
}

__global__ void multiFlipSide(Point2d* points, int2* edge2face, int2* edges, int2* edgeSide, int4* edge2edge, int* atomicFace, int* canFlip, int* flipId, int* needTest, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= canFlip[numE]) return;
	int otid = tid;
	tid = flipId[tid];

	int4 e2e = edge2edge[tid];
	int2 e2f = edge2face[tid];
	int2 es = edgeSide[tid];
	flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, es.x, es.y, e2f, e2e, tid);
	int p1 = es.x;
	int p2 = es.y;
	Point2d P1, Px, Py, P2;
	Py = points[p1]; Px = points[p2];
	int2 ev = edges[tid];
	P1 = points[ev.x]; P2 = points[ev.y];
	flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, p1, p2, tid);
	int py = p1, px = p2;
	p1 = ev.x; p2 = ev.y;
	bool change = true;
	int eid = tid;
	int c = 1;
	while (change) {
		//printf("%d: %d %d %d %d\n", tid, px, py, p1, p2);
		needTest[eid] = 0;
		needTest[e2e.x / 2] = -1;
		needTest[e2e.y / 2] = -1;
		needTest[e2e.z / 2] = -1;
		needTest[e2e.w / 2] = -1;
		change = false;
		int fid = getInt2(edge2face + e2e.x / 2, 1 - (e2e.x % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int nx = getInt2(edgeSide + e2e.x / 2, (1 - e2e.x % 2));
			Point2d Pn = points[nx];
			if (inCircle(Pn, P1, Py, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//printf("flip x\n");
					//FLIP!
					eid = e2e.x / 2;
					int np1, np2;
					if (e2e.x % 2) {
						np1 = nx;
						np2 = px;

						p2 = py;
						py = nx;
						P2 = Py;
						Py = Pn;
					}
					else {
						np1 = px;
						np2 = nx;

						p2 = p1;
						p1 = py;
						py = px;
						px = nx;
						P2 = P1;
						P1 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.x % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.x % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);
					c++;
					change = true;

					continue;
				}
			}
		}
		fid = getInt2(edge2face + e2e.y / 2, 1 - (e2e.y % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int ny = getInt2(edgeSide + e2e.y / 2, (1 - e2e.y % 2));
			Point2d Pn = points[ny];
			if (inCircle(Pn, Px, P1, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.y / 2;
					//printf("flip y\n");
					int np1, np2;
					if (e2e.y % 2) {
						np1 = ny;
						np2 = py;

						p2 = p1;
						p1 = px;
						px = py;
						py = ny;
						P2 = P1;
						P1 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = py;
						np2 = ny;

						p2 = px;
						px = ny;
						P2 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.y % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.y % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.z / 2, 1 - (e2e.z % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int nz = getInt2(edgeSide + e2e.z / 2, (1 - e2e.z % 2));
			Point2d Pn = points[nz];
			if (inCircle(Pn, P2, Px, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.z / 2;
					//printf("flip z\n");
					int np1, np2;
					if (e2e.z % 2) {
						np1 = nz;
						np2 = py;

						p1 = p2;
						p2 = px;
						px = py;
						py = nz;
						P1 = P2;
						P2 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = py;
						np2 = nz;

						p1 = px;
						px = nz;
						P1 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.z % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.z % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.w / 2, 1 - (e2e.w % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int nw = getInt2(edgeSide + e2e.w / 2, (1 - e2e.w % 2));
			Point2d Pn = points[nw];
			if (inCircle(Pn, Py, P2, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					//printf("flip w\n");
					eid = e2e.w / 2;
					int np1, np2;
					if (e2e.w % 2) {
						np1 = nw;
						np2 = px;

						p1 = py;
						py = nw;
						P1 = Py;
						Py = Pn;
					}
					else {
						np1 = px;
						np2 = nw;

						p1 = p2;
						p2 = py;
						py = px;
						px = nw;

						P1 = P2;
						P2 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.w % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.w % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}
	}

	change = true;
	e2e = edge2edge[tid];
	e2f = edge2face[tid];
	eid = tid;
	ev = edges[tid];
	p1 = getInt2(edges + e2e.y / 2, e2e.y % 2);
	p2 = getInt2(edges + e2e.w / 2, e2e.w % 2);
	P1 = points[p1]; P2 = points[p2];
	Px = points[ev.x]; Py = points[ev.y];
	py = ev.y, px = ev.x;
	bool first = true;
	while (change)
	{
		//printf("%d: %d %d %d %d\n", tid, px, py, p1, p2);
		if (first) {
			first = false;
		}
		else {
			needTest[eid] = 0;
			needTest[e2e.x / 2] = -1;
			needTest[e2e.y / 2] = -1;
			needTest[e2e.z / 2] = -1;
			needTest[e2e.w / 2] = -1;
		}
		change = false;

		int fid = getInt2(edge2face + e2e.z / 2, 1 - (e2e.z % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int nz = getInt2(edgeSide + e2e.z / 2, (1 - e2e.z % 2));
			Point2d Pn = points[nz];
			if (inCircle(Pn, P2, Px, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.z / 2;
					//printf("flip z\n");
					int np1, np2;
					if (e2e.z % 2) {
						np1 = nz;
						np2 = py;

						p1 = p2;
						p2 = px;
						px = py;
						py = nz;
						P1 = P2;
						P2 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = py;
						np2 = nz;

						p1 = px;
						px = nz;
						P1 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.z % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.z % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.w / 2, 1 - (e2e.w % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int nw = getInt2(edgeSide + e2e.w / 2, (1 - e2e.w % 2));
			Point2d Pn = points[nw];
			if (inCircle(Pn, Py, P2, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					//printf("flip w\n");
					eid = e2e.w / 2;
					int np1, np2;
					if (e2e.w % 2) {
						np1 = nw;
						np2 = px;

						p1 = py;
						py = nw;
						P1 = Py;
						Py = Pn;
					}
					else {
						np1 = px;
						np2 = nw;

						p1 = p2;
						p2 = py;
						py = px;
						px = nw;

						P1 = P2;
						P2 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.w % 2) = e2f.y;
					getInt2(&e2f, 1 - (e2e.w % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}

		fid = getInt2(edge2face + e2e.x / 2, 1 - (e2e.x % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int nx = getInt2(edgeSide + e2e.x / 2, (1 - e2e.x % 2));
			Point2d Pn = points[nx];
			if (inCircle(Pn, P1, Py, Px) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//printf("flip x\n");
					//FLIP!
					eid = e2e.x / 2;
					int np1, np2;
					if (e2e.x % 2) {
						np1 = nx;
						np2 = px;

						p2 = py;
						py = nx;
						P2 = Py;
						Py = Pn;
					}
					else {
						np1 = px;
						np2 = nx;

						p2 = p1;
						p1 = py;
						py = px;
						px = nx;
						P2 = P1;
						P1 = Py;
						Py = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.x % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.x % 2)) = fid;
					e2e = edge2edge[eid];
					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);
					c++;
					change = true;

					continue;
				}
			}
		}
		fid = getInt2(edge2face + e2e.y / 2, 1 - (e2e.y % 2));
		if (fid != -1 && atomicFace[fid] != -2) {
			int ny = getInt2(edgeSide + e2e.y / 2, (1 - e2e.y % 2));
			Point2d Pn = points[ny];
			if (inCircle(Pn, Px, P1, Py) > 0)
			{
				//int lock = atomicCAS(atomicFace + fid, -1, tid);
				int lock = atomicExch(atomicFace + fid, -2);
				if (lock != -2)
				{
					//FLIP!
					eid = e2e.y / 2;
					//printf("flip y\n");
					int np1, np2;
					if (e2e.y % 2) {
						np1 = ny;
						np2 = py;

						p2 = p1;
						p1 = px;
						px = py;
						py = ny;
						P2 = P1;
						P1 = Px;
						Px = Py;
						Py = Pn;
					}
					else {
						np1 = py;
						np2 = ny;

						p2 = px;
						px = ny;
						P2 = Px;
						Px = Pn;
					}
					getInt2(&e2f, e2e.y % 2) = e2f.x;
					getInt2(&e2f, 1 - (e2e.y % 2)) = fid;
					e2e = edge2edge[eid];

					ev = edges[eid];
					flipEdgeNeighbor(edge2face, edges, edgeSide, edge2edge, np1, np2, e2f, e2e, eid);
					flipEdgeSelf(edges, edgeSide, edge2edge, e2e, ev.x, ev.y, np1, np2, eid);

					change = true;
					c++;
					continue;
				}
			}
		}
	}

	canFlip[otid] = c;
}

__global__ void compactFlip(int* canFlip, int* flipId, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	if (canFlip[tid] + 1 == canFlip[tid + 1])
	{
		flipId[canFlip[tid]] = tid;
	}
}

__global__ void makeEdgeWithPoint(Point2d* points, int2* edges, Point2d2* edgeWithPoint, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;
	int2 ev = edges[tid];
	Point2d2 ret;
	ret.x = points[ev.x];
	ret.y = points[ev.y];
	edgeWithPoint[tid] = ret;
}

__global__ void updateFace2EdgeAndEdge2Face(int3* tris, int2* edges, int2* edge2face, int3* face2edge, int numE)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numE) return;

	int2 ef = edge2face[tid];
	int2 ev = edges[tid];
	if (ef.x != -1)
	{
		int3 t = tris[ef.x];
		int other = t.x ^ t.y ^ t.z ^ ev.x ^ ev.y;
		int id = getIdInt3(t, other);
		getInt3(face2edge + ef.x, id) = tid * 2;
		ef.x = ef.x * 3 + id;
	}
	if (ef.y != -1)
	{
		int3 t = tris[ef.y];
		int other = t.x ^ t.y ^ t.z ^ ev.x ^ ev.y;
		int id = getIdInt3(t, other);
		getInt3(face2edge + ef.y, id) = tid * 2 + 1;
		ef.y = ef.y * 3 + id;
	}
	edge2face[tid] = ef;
}

double delaunayNew(void* cubTemp, size_t cubBytes, Point2d*& points, int3*& tris, int2* cons, int numPoints, int numTriangles, int numCons)
{
	double costTime = 0;
	costTime += reorderPoints(cubTemp, cubBytes, points, tris, cons, numPoints, numTriangles, numCons);
	costTime += reorderFaces(cubTemp, cubBytes, points, tris, numTriangles);
	int2* edges;
	
	int2* edge2face;
	int numEdges;
	int* noFlips;
	double edgetime = updateEdgeRadixSort_new(cubTemp, cubBytes, tris, edges, edge2face, cons, noFlips, numCons, numEdges, numPoints, numTriangles);
	printf("numEdge: %d\n", numEdges);
	//reorderEdge(cubTemp, cubBytes, points, edges, edge2face, numTriangles, numEdges);

	//int4* edge2edge;
	//int2* edgeSide;
	//costTime += updateEdgeToEdge(cubTemp, cubBytes, tris, edges, edgeSide, edge2face, edge2edge, numEdges, numTriangles);
	int3* face2edge;
	utils::malloc(face2edge, numEdges);
	costTime += utils::kernel("update fe ef", updateFace2EdgeAndEdge2Face, numEdges, 128, tris, edges, edge2face, face2edge, numEdges);
	//unsigned int* randSeeds;
	//utils::malloc(randSeeds, numEdges);
	//utils::kernel(initSeed, numEdges, 128, randSeeds, numEdges);
	
	int* needTest;
	utils::malloc(needTest, numEdges + 1);
	int* canFlip;
	utils::malloc(canFlip, numEdges + 1);
	int* needFlip;
	utils::malloc(needFlip, numEdges + 1);
	int* flipId;
	utils::malloc(flipId, numEdges);

	int* flipped;
	utils::malloc(flipped, numTriangles);
	int* atomicFace;
	utils::malloc(atomicFace, numTriangles);
	int* queue;
	utils::malloc(queue, numPoints * QUEUE_SIZE);
	int* d_mark;
	utils::malloc(d_mark, 1);
	int mark = 1;
	int time = 0;
	printf("start\n");
	Point2d2* edgesWithPoint;
	utils::malloc(edgesWithPoint, numEdges);
	{
		//int3* h_tris = new int3[numTriangles];
		//utils::memcpy(h_tris, tris, numTriangles, cudaMemcpyDeviceToHost);
		//int* count = new int[numTriangles];
		//memset(count, 0, numTriangles * sizeof(int));
		//for (int i = 0; i < numTriangles; i++)
		//{
		//	//if (h_tris[i].x == 6 || h_tris[i].y == 6 || h_tris[i].z == 6)
		//	//{
		//		printf("%d %d %d %d\n", i, h_tris[i].x, h_tris[i].y, h_tris[i].z);
		//	//	count[i] = 1;
		//	//}
		//}
		//int2* h_edge = new int2[numEdges];
		//utils::memcpy(h_edge, edge2face, numEdges, cudaMemcpyDeviceToHost);
		//for (int i = 0; i < numEdges; i++)
		//{
		//	printf("%d %d %d\n", i, h_edge[i].x, h_edge[i].y);
		//}

		cudaEvent_t start, stop;
		float elapsedTime = 0.0;
		float kernelTime = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		utils::memset(d_mark, 1);
		utils::memset(atomicFace, numTriangles, -1);
		utils::memset(needTest, numEdges, -1);
		utils::kernel(initNeedFlipFE_new, numEdges, 128, points, edges, edge2face, face2edge, tris, needTest, needFlip, noFlips, atomicFace, d_mark, numEdges);
		//utils::kernel("init", initNeedFlipFE_new, numEdges, 128, points, edges, edge2face, face2edge, tris, needTest, needFlip, atomicFace, d_mark, numEdges);
		//utils::kernel(makeEdgeWithPoint, numEdges, 128, points, edges, edgesWithPoint, numEdges);
		//utils::kernel(initNeedFlip_new, numEdges, 128, points, edgesWithPoint, edge2face, edge2edge, needTest, needFlip, atomicFace, d_mark, numEdges);
		mark = utils::getValue(d_mark, 0);
		//int tt = 10;
		double confirm, compact, multi, init;
		confirm = compact = multi = init = 0.0;
		//int tt = 20;
		while (mark) {
			time++;
			//utils::memset(flipped, numTriangles, -1);
			utils::kernel(confirmFlip, numEdges, 128, edge2face, atomicFace, needFlip, canFlip, flipped, numEdges);
			utils::exlusiveScan(cubTemp, cubBytes, canFlip, canFlip, numEdges + 1);
			utils::kernel(compactFlip, numEdges, 128, canFlip, flipId, numEdges);
			int threads = utils::getValue(canFlip, numEdges);
			utils::kernel(multiFlipFE, threads, 32, points, edge2face, face2edge, tris, atomicFace, canFlip, flipId, needTest, noFlips, numEdges);
			//utils::kernel(multiFlip, threads, 128, points, edge2face, edgesWithPoint, edge2edge, atomicFace, canFlip, flipId, needTest, numEdges);
			//utils::kernel(multiFlipSide, threads, 128, points, edge2face, edges, edgeSide, edge2edge, atomicFace, canFlip, flipId, needTest, numEdges);
			//utils::kernel(multiFlipWithStack, threads, 128, points, edge2face, edges, edge2edge, flipped, canFlip, flipId, needTest, queue, numEdges);
			//utils::exlusiveScan(cubTemp, cubBytes, canFlip, canFlip, threads + 1);
			//int sum = utils::getValue(canFlip, threads);
			//printf("sum %d %d\n", threads, sum);
			utils::memset(d_mark, 1);
			utils::memset(atomicFace, numTriangles, -1);
			//utils::memset(needTest, numEdges, -1);
			utils::kernel(initNeedFlipFE_new, numEdges, 128, points, edges, edge2face, face2edge, tris, needTest, needFlip, noFlips, atomicFace, d_mark, numEdges);
			//utils::kernel(initNeedFlip_new, numEdges, 128, points, edgesWithPoint, edge2face, edge2edge, needTest, needFlip, atomicFace, d_mark, numEdges);
			
			//int3* h_tris = new int3[numTriangles];
			//utils::memcpy(h_tris, tris, numTriangles, cudaMemcpyDeviceToHost);
			//int* count = new int[numTriangles];
			//memset(count, 0, numTriangles * sizeof(int));
			//for (int i = 0; i < numTriangles; i++)
			//{
			//	if (h_tris[i].x == 6 || h_tris[i].y == 6 || h_tris[i].z == 6)
			//	{
			//		printf("%d %d %d %d\n", i, h_tris[i].x, h_tris[i].y, h_tris[i].z);
			//		count[i] = 1;
			//	}
			//}
			//int2* h_edge = new int2[numEdges];
			//utils::memcpy(h_edge, edge2face, numEdges, cudaMemcpyDeviceToHost);
			//for (int i = 0; i < numEdges; i++)
			//{
			//	if (count[h_edge[i].x] == 1 && count[h_edge[i].y] == 1)
			//	{
			//		printf("%d %d %d\n", i, h_edge[i].x, h_edge[i].y);
			//	}
			//}
			mark = utils::getValue(d_mark, 0);
			//testTopo(edge2face, edges, edge2edge, numEdges);
		}
		//utils::memset(atomicFace, numTriangles);
		//utils::kernel(rebuildFaceInfo, numEdges, 128, tris, edges, edgeSide, edge2face, atomicFace, numEdges);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		std::cout << "flip times: " << time << std::endl;
		std::cout << "kernel" << " time: " << kernelTime << " ms\n";
		std::cout << "flipEdge" << " time: " << elapsedTime << " ms\n";
		printf("%f %f %f %f\n", confirm,compact,multi,init);
		costTime += elapsedTime;

		utils::memset(d_mark, 1);
		utils::memset(needTest, numEdges, -1);
		utils::kernel("init", initNeedFlipFE_new, numEdges, 128, points, edges, edge2face, face2edge, tris, needTest, needFlip, noFlips, atomicFace, d_mark, numEdges);
		//utils::kernel(initNeedFlip_new, numEdges, 128, points, edgesWithPoint, edge2face, edge2edge, needTest, needFlip, atomicFace, d_mark, numEdges);
		mark = utils::getValue(d_mark, 0);
		printf("mark %d costTime %f %f\n", mark, costTime, costTime+edgetime);
		costTime += edgetime;
	}
	//drawEdge(points, edges, numPoints, numEdges);
	utils::release(canFlip, numEdges + 1);
	utils::release(needFlip, numEdges + 1);
	utils::release(atomicFace, numTriangles);
	utils::release(flipped, numTriangles);
	utils::release(edges, (int)numTriangles * 2);
	utils::release(edge2face, (int)numTriangles * 2);
	utils::release(d_mark, 1);
	utils::release(flipId, numEdges);
	utils::release(needTest, numEdges + 1);
	utils::release(queue, numPoints * QUEUE_SIZE * 2);
	printf("%f\n", costTime);
	return costTime;
}
double delaunay(void* cubTemp, size_t cubBytes, Point2d*& points, int3*& tris, int numPoints, int numTriangles)
{
	double costTime = 0;
	costTime += reorderPoints(cubTemp, cubBytes, points, tris, nullptr, numPoints, numTriangles, 0);
	costTime += reorderFaces(cubTemp, cubBytes, points, tris, numTriangles);
	int2* edges;
	int2* edge2face;
	int2* edgeSide;
	int numEdges;
	int* noFlips;
	costTime += updateEdgeRadixSort_new(cubTemp, cubBytes, tris, edges, edge2face, nullptr, noFlips, 0, numEdges, numPoints, numTriangles);
	printf("numEdge: %d\n", numEdges);
	//reorderEdge(cubTemp, cubBytes, points, edges, edge2face, numTriangles, numEdges);

	int4* edge2edge;
	costTime += updateEdgeToEdge(cubTemp, cubBytes, tris, edges, edgeSide, edge2face, edge2edge, numEdges, numTriangles);
	utils::malloc(edgeSide, numEdges);

	//{
	//	int3* h_tris = new int3[numTriangles];
	//	utils::memcpy(h_tris, tris, numTriangles, cudaMemcpyDeviceToHost);
	//
	//	std::map<std::tuple<int, int>, int> edgeMap;
	//	int2* h_edges = new int2[numTriangles * 3];
	//	int2* h_edge2face = new int2[numTriangles * 3];
	//	memset(h_edge2face, -1, sizeof(int2) * numTriangles * 3);
	//	int idx = 0;
	//	for (int i = 0; i < numTriangles; i++)
	//	{
	//		int3 t = h_tris[i];
	//		std::tuple<int, int> key;
	//		if (t.x < t.y)
	//			key = std::make_tuple(t.x, t.y);
	//		else
	//			key = std::make_tuple(t.y, t.x);
	//		if (edgeMap.find(key) == edgeMap.end())
	//		{
	//			h_edges[idx] = make_int2(std::get<0>(key), std::get<1>(key));
	//			h_edge2face[idx].x = i;
	//			edgeMap[key] = idx;
	//			idx++;
	//		}
	//		else {
	//			int id = edgeMap[key];
	//			h_edge2face[id].y = i;
	//		}
	//
	//		if (t.y < t.z)
	//			key = std::make_tuple(t.y, t.z);
	//		else
	//			key = std::make_tuple(t.z, t.y);
	//
	//		if (edgeMap.find(key) == edgeMap.end())
	//		{
	//			h_edges[idx] = make_int2(std::get<0>(key), std::get<1>(key));
	//			h_edge2face[idx].x = i;
	//			edgeMap[key] = idx;
	//			idx++;
	//		}
	//		else {
	//			int id = edgeMap[key];
	//			h_edge2face[id].y = i;
	//		}
	//
	//		if (t.x < t.z)
	//			key = std::make_tuple(t.x, t.z);
	//		else
	//			key = std::make_tuple(t.z, t.x);
	//
	//		if (edgeMap.find(key) == edgeMap.end())
	//		{
	//			h_edges[idx] = make_int2(std::get<0>(key), std::get<1>(key));
	//			h_edge2face[idx].x = i;
	//			edgeMap[key] = idx;
	//			idx++;
	//		}
	//		else {
	//			int id = edgeMap[key];
	//			h_edge2face[id].y = i;
	//		}
	//	}
	//
	//	numEdges = idx;
	//	printf("number of edge %d\n", numEdges);
	//
	//	utils::malloc(edges, numEdges);
	//	utils::memcpy(edges, h_edges, numEdges);
	//	utils::malloc(edge2face, numEdges);
	//	utils::memcpy(edge2face, h_edge2face, numEdges);
	//
	//	delete[] h_edges;
	//	delete[] h_edge2face;
	//	delete[] h_tris;
	//}

	unsigned int* randSeeds;
	//utils::malloc(randSeeds, numEdges);
	//utils::kernel(initSeed, numEdges, 128, randSeeds, numEdges);

	//int* canFlip;
	//utils::malloc(canFlip, numEdges);
	int* needFlip;
	int* sumFlip;
	utils::malloc(needFlip, numEdges + 1);
	utils::malloc(sumFlip, numEdges + 1);
	utils::memset(needFlip, numEdges + 1);

	//int* needTest;
	//utils::malloc(needTest, numEdges + 1);
	//int* needTestOffset;
	//utils::malloc(needTestOffset, numEdges + 1);

	int* atomicFace;
	utils::malloc(atomicFace, numTriangles);
	int* futureFace;
	utils::malloc(futureFace, numTriangles);
	int* d_mark;
	utils::malloc(d_mark, 1);
	int mark = 1;
	int time = 0;
	printf("start\n");
	{
		cudaEvent_t start, stop;
		float elapsedTime = 0.0;
		float kernelTime = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		utils::kernel(rebuildEdgeInfo, numEdges, 128, tris, edges, edgeSide, edge2face, numEdges);
		//utils::kernel(clearNeedTest, numEdges, 128, needTest, numEdges);
		utils::memset(d_mark, 1);
		utils::memset(atomicFace, numTriangles, -1);
		//utils::kernel(initNeedFlip, numEdges, 128, points, tris, edge2face, needFlip, atomicFace, d_mark, numEdges);
		utils::kernel(initNeedFlip_n, numEdges, 128, points, edges, edgeSide, edge2face, needFlip, atomicFace, d_mark, numEdges);
		mark = utils::getValue(d_mark, 0);
		while (mark != 0) {
			time++;
			utils::memset(futureFace, numTriangles, -1);
			//utils::kernel(flipEdge_testAndflip, numEdges, 128, tris, edges, edge2face, needFlip, atomicFace, futureFace, numEdges);
			utils::kernel(flipEdge_testAndflip_n, numEdges, 128, edges, edgeSide, edge2face, needFlip, atomicFace, futureFace, numEdges);
			utils::memset(atomicFace, numTriangles, -1);
			utils::memset(d_mark, 1);
			//utils::kernel(flipEdge_healAndtest, numEdges, 128, points, tris, edges, edge2face, futureFace, needFlip, d_mark, atomicFace, numEdges);
			utils::kernel(flipEdge_healAndtest_n, numEdges, 128, points, edges, edgeSide, edge2face, futureFace, needFlip, d_mark, atomicFace, numEdges);
			mark = utils::getValue(d_mark, 0);
			//utils::exlusiveScan(cubTemp, cubBytes, needFlip, sumFlip, numEdges + 1);
			//int sum = utils::getValue(sumFlip, numEdges);
			//printf("sum: %d\n", sum);
			//draw(points, tris, canFlip, needTest, numPoints, numTriangles);
			//cudaDeviceSynchronize();
		}
		utils::memset(atomicFace, numTriangles);
		utils::kernel(rebuildFaceInfo, numEdges, 128, tris, edges, edgeSide, edge2face, atomicFace, numEdges);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		std::cout << "flip times: " << time << std::endl;
		std::cout << "kernel" << " time: " << kernelTime << " ms\n";
		std::cout << "flipEdge" << " time: " << elapsedTime << " ms\n";
		costTime += elapsedTime;
	}
	//drawEdge(points, edges, numPoints, numEdges);
	utils::release(randSeeds, numEdges);
	//utils::release(canFlip, numEdges);
	utils::release(needFlip, numEdges + 1);
	utils::release(atomicFace, numTriangles);
	utils::release(futureFace, numTriangles);
	utils::release(edgeSide, numEdges);
	utils::release(edges, (int)numTriangles * 2);
	utils::release(edge2face, (int)numTriangles * 2);
	utils::release(d_mark, 1);
	utils::release(sumFlip, numEdges + 1);
	return costTime;
}
double delaunaySample(void* cubTemp, size_t cubBytes, Point2d* points, int3* tris, int3* adjTris, int numPoints, int numTriangles)
{
	int2* edges;
	int2* edge2face;
	int2* edgeSide;
	double costTime = 0;
	int numEdges;
	int2* cons;
	int* noFlips;
	costTime += updateEdgeRadixSort_new(cubTemp, cubBytes, tris, edges, edge2face, cons, noFlips, 0, numEdges, numPoints, numTriangles);
	printf("numEdge: %d\n", numEdges);
	//reorderEdge(cubTemp, cubBytes, points, edges, edge2face, numEdges);
	utils::malloc(edgeSide, numEdges);

	int* needFlip;
	utils::malloc(needFlip, numEdges);

	int* atomicFace;
	utils::malloc(atomicFace, numTriangles);
	int* futureFace;
	utils::malloc(futureFace, numTriangles);
	int* d_mark;
	utils::malloc(d_mark, 1);
	int mark = 1;
	int time = 0;
	printf("start\n");
	{
		cudaEvent_t start, stop;
		float elapsedTime = 0.0;
		float kernelTime = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		utils::kernel(rebuildEdgeInfo, numEdges, 128, tris, edges, edgeSide, edge2face, numEdges);
		utils::memset(d_mark, 1);
		utils::memset(atomicFace, numTriangles, -1);
		utils::kernel(initNeedFlip_n, numEdges, 128, points, edges, edgeSide, edge2face, needFlip, atomicFace, d_mark, numEdges);
		mark = utils::getValue(d_mark, 0);
		while (mark != 0) {
			time++;
			utils::memset(futureFace, numTriangles, -1);
			utils::kernel(flipEdge_testAndflip_n, numEdges, 128, edges, edgeSide, edge2face, needFlip, atomicFace, futureFace, numEdges);
			utils::memset(atomicFace, numTriangles, -1);
			utils::memset(d_mark, 1);
			utils::kernel(flipEdge_healAndtest_n, numEdges, 128, points, edges, edgeSide, edge2face, futureFace, needFlip, d_mark, atomicFace, numEdges);
			mark = utils::getValue(d_mark, 0);
		}
		utils::memset(atomicFace, numTriangles);
		utils::kernel(rebuildFaceInfo, numEdges, 128, tris, edges, edgeSide, edge2face, atomicFace, numEdges);
		utils::kernel(rebuildFaceAdjInfo, numEdges, 128, tris, adjTris, edgeSide, edge2face, numEdges);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		std::cout << "flip times: " << time << std::endl;
		std::cout << "kernel" << " time: " << kernelTime << " ms\n";
		std::cout << "flipEdge" << " time: " << elapsedTime << " ms\n";
		costTime += elapsedTime;
	}

	utils::release(needFlip, numEdges);
	utils::release(atomicFace, numTriangles);
	utils::release(futureFace, numTriangles);
	utils::release(edge2face, numEdges);
	utils::release(edges, numEdges);
	utils::release(edgeSide, numEdges);
	utils::release(d_mark, 1);
	return costTime;
}
void updateAdj(int3* tris, int3* adjTris, int numPoints, int numTriangles)
{
	utils::kernel(clearAdjTris, numTriangles, 128, adjTris, numTriangles);
	
	size_t hashSize = numPoints * 2;
	int* hashCount;
	int3* hashValue;
	utils::malloc(hashCount, hashSize);
	utils::malloc(hashValue, hashSize);
	
	
	int* mark;
	utils::malloc(mark, 1);
	
	int d_mark = 4;
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	utils::memset(mark, 1);
	utils::memset(hashCount, hashSize);
	while (d_mark != 0){
		utils::kernel("s1", remakeAdjFace_s1, numTriangles, 128, tris, adjTris, hashCount, hashValue, mark, numTriangles);
		utils::kernel("s2", remakeAdjFace_s2, numTriangles, 128, tris, adjTris, hashCount, hashValue, numTriangles);
		utils::kernel("s3", remakeAdjFace_s3, hashSize, 128, tris, adjTris, hashCount, hashValue, (int)hashSize);
		d_mark = utils::getValue(mark, 0);
		utils::memset(mark, 1);
		printf("%d\n", d_mark);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "update adj" << " time: " << elapsedTime << " ms\n";

	utils::release(hashCount, hashSize);
	utils::release(hashValue, hashSize);
	utils::release(mark, 1);
}

bool operator <(const Point2d& a, const Point2d& b)
{
	return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
void readFile(const std::string& filename, double*& points, int& numPoints, int2*& cons, int& numCons, bool shuffle = false)
{
	std::ios::sync_with_stdio(false);
	std::ifstream infile;
	infile.open(filename, std::ios::in);
	if (!infile)
	{
		printf("open file failed\n");
	}
	infile >> numPoints;
	printf("numPoinst %d\n", numPoints);
	points = new double[numPoints * 2];
	double minx = 1000000, miny = 1000000, maxx = -1000000, maxy = -1000000;

	std::map<std::pair<double, double>, int> pmap;
	int* idxmap = new int[numPoints];
	int ridx = 0;
	int remove = 0;
	for (int i = 0; i < numPoints; i++)
	{
		int idx;
		infile >> idx;
		double x, y;
		infile >> x >> y;
		//points[idx * 2] = x;
		//points[idx * 2 + 1] = y;
		auto p = std::make_pair(x, y);
		if (pmap.find(p) == pmap.end())
		{
			pmap[p] = ridx;
			points[ridx * 2] = x;
			points[ridx * 2 + 1] = y;
			idxmap[idx] = ridx;
			ridx++;
			if (x > maxx) maxx = x;
			if (x < minx) minx = x;
			if (y > maxy) maxy = y;
			if (y < miny) miny = y;
		}
		else {
			//printf("remove\n");
			remove++;
			idxmap[idx] = pmap[p];
		}
	}
	printf("remove %d\n", remove);	
	double cx = (minx + maxx) / 2;
	double cy = (miny + maxy) / 2;
	double lx = maxx - minx;
	double ly = maxy - miny;
	if (ly > lx) lx = ly;
	numPoints = ridx;
	printf("points %d\n", numPoints);

	int* random;
	if (shuffle)
	{
		random = new int[numPoints];
		for (int i = 0; i < numPoints; i++)
		{
			random[i] = i;
		}
		std::mt19937 gen(19999);
		std::shuffle(random, random + numPoints, gen);
		double* nPoints = new double[numPoints * 2];
		for (int i = 0; i < numPoints; i++)
		{
			int nid = random[i];
			nPoints[nid * 2] = points[i * 2];
			nPoints[nid * 2 + 1] = points[i * 2 + 1];
		}
		delete[] points;
		points = nPoints;
	}

	for (int i = 0; i < numPoints; i++)
	{
		points[i * 2] -= cx;
		points[i * 2] = points[i * 2] / lx * 0.99;
		points[i * 2] += 0.5;
	
		points[i * 2 + 1] -= cy;
		points[i * 2 + 1] = points[i * 2 + 1] / lx * 0.99;
		points[i * 2 + 1] += 0.5;
	}

	infile >> numCons;
	printf("cons %d\n", numCons);
	
	if (numCons != 0) {
		cons = new int2[numCons];
		std::set<std::pair<int, int>> cmap;
		int idx = 0;
		for (int i = 0; i < numCons; i++)
		{
			int temp;
			infile >> temp;
			int x, y;
			infile >> x >> y;
			if (shuffle)
			{
				x = random[idxmap[x]];
				y = random[idxmap[y]];
			}
			else {
				x = idxmap[x];
				y = idxmap[y];
			}
			if (x == y || cmap.find(std::make_pair(x, y)) != cmap.end() || cmap.find(std::make_pair(y, x)) != cmap.end())
			{
				//printf("ign %d %d\n", x, y);
			}
			else {
				cons[idx].x = x;
				cons[idx].y = y;
				cmap.insert(std::make_pair(x, y));
				//if (i < 100) printf("ins %d %d\n", x, y);
				idx++;
			}
		}
		numCons = idx;
	}
	else {
		cons = nullptr;
	}
	printf("cons %d\n", numCons);
	delete[] idxmap;
	if (shuffle)
	{
		delete[] random;
	}
	infile.close();
}

double filterTriangles(void* cubTemp, size_t cubBytes, int3* tris, int3* newTris, int3* adjTris, int3* newAdjTris, int* sons, int& numTriangles)
{
	int* triCounts;
	utils::malloc(triCounts, numTriangles + 1);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::kernel(filterTriangles_s1, numTriangles, 128, sons, triCounts, numTriangles);
	utils::exlusiveScan(cubTemp, cubBytes, triCounts, triCounts, numTriangles + 1);

	if (adjTris != nullptr) {
		utils::kernel(filterTriangles_s2_new, numTriangles, 128, tris, newTris, adjTris, newAdjTris, sons, triCounts, numTriangles);
	}
	else {
		utils::kernel(filterTriangles_s2, numTriangles, 128, tris, newTris, sons, triCounts, numTriangles);
	}
	numTriangles = utils::getValue(triCounts, numTriangles);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "filter" << " time: " << ' ' << elapsedTime << " ms\n";

	utils::release(triCounts, numTriangles + 1);
	
	printf("filter triangles num %d\n", numTriangles);
	return elapsedTime;
}

void drawFinal(Point2d* points, int3* tris, int2* d_cons, int numPoints, int numTriangles, int numCons)
{
}

#include <iomanip>

int main(int argc, char* argv[])
{
	getMemCacheRef().initAbtrtMem();
	unsigned char* cubTempStorage = nullptr;
	size_t cubTempStorageBytes = 0;

	//int numPoints, numCons;
	//double* h_fpoints;
	//int2* h_cons;
	//readFile(argv[1], h_fpoints, numPoints, h_cons, numCons, false);
	//numCons = 0;

	int numPoints = 9000000;
	int numCons = 0;
	int2* h_cons;
	if (numCons != 0) {
		h_cons = new int2[numCons];
		h_cons[0] = make_int2(0, 1);
		h_cons[1] = make_int2(1, 2);
		h_cons[2] = make_int2(2, 3);
	}
	double* h_fpoints = new double[numPoints * 2];
	randomMap(h_fpoints, numPoints, 13159);
	//thinCircleMap(h_fpoints, numPoints, 13159);
	//gaussianMap(h_fpoints, numPoints, 13159);
	//twoLineMap(h_fpoints, numPoints, 13159);
	//squareMap(h_fpoints, numPoints, 13156);

	const int maxTriangles = numPoints * 4;
	printf("%d\n", maxTriangles);
	double* fpoints;
	utils::mallocAndCpy(fpoints, h_fpoints, numPoints * 2);

	see << <1, 1 >> > (fpoints);

	Point2d* points;
	utils::malloc(points, numPoints + 4);
	MyInteger<logPointRes> res;
	{
		assert(logPointRes < 63);
		long long tempRes = 1LL << logPointRes;
		utils::kernel("transPoints", transPoints, numPoints, 128, points, fpoints, tempRes, numPoints);
		res = tempRes - 1;
	}
	utils::release(fpoints, numPoints * 2);

	Point2d* h_points = new Point2d[numPoints];
	utils::memcpy(h_points, points, numPoints, cudaMemcpyDeviceToHost);

	delete[] h_fpoints;
	//utils::exlusiveScanRegist(countNewTriangles, countNewTriangles, maxTriangles, cubTempStorageBytes);
	//utils::exlusiveScanRegist(overlapCounts, overlapOffsets, maxNumTh, cubTempStorageBytes);

	printf("cub %d\n", cubTempStorageBytes);
	cubTempStorageBytes = 983743231;
	utils::malloc(cubTempStorage, cubTempStorageBytes);

	//reorderPoints(cubTempStorage, cubTempStorageBytes, points, nullptr, numPoints, 0);
	int3* tris, * adjTris;
	int* sons;
	int numTriangles;
	int3* newTris;
	double costTime = 0;
	if (true) {
		Point2d* samplePoints;
		int* sampleId;
		int numSample;
		costTime += getSamplePoints(cubTempStorage, cubTempStorageBytes, points, numPoints, samplePoints, sampleId, numSample);
		printf("%f\n", costTime);
		utils::malloc(tris, maxTriangles);
		utils::malloc(adjTris, maxTriangles);
		utils::malloc(sons, maxTriangles);
		costTime += rawMeshBefore(cubTempStorage, cubTempStorageBytes, samplePoints, tris, sons, numTriangles, numSample);
		printf("%f\n", costTime);
		int3* newTrisSample;
		utils::malloc(newTrisSample, numTriangles);
		costTime += filterTriangles(cubTempStorage, cubTempStorageBytes, tris, newTrisSample, nullptr, nullptr, sons, numTriangles);
		printf("%f\n", costTime);
		costTime += delaunaySample(cubTempStorage, cubTempStorageBytes, samplePoints, newTrisSample, adjTris, numSample + 4, numTriangles);
		printf("%f\n", costTime);
		//drawFinal(samplePoints, newTrisSample, nullptr, numSample + 4, numTriangles, 0);
		int* bitmapOffsets, * bitmapTris;
		costTime += getBitmap(cubTempStorage, cubTempStorageBytes, samplePoints, newTrisSample, bitmapOffsets, bitmapTris, numTriangles);
		//drawFinal(samplePoints, newTrisSample, numSample + 4, numTriangles);
		printf("%f\n", costTime);
		costTime += utils::kernel("convertSample", convertSample, numTriangles, 128, newTrisSample, sampleId, numSample, numPoints, numTriangles);
		utils::memcpy(tris, newTrisSample, numTriangles, cudaMemcpyDeviceToDevice);
		printf("%d %d\n", numSample, numTriangles);
		//printf();
		costTime += rawMeshAfter(cubTempStorage, cubTempStorageBytes, points, tris, adjTris, sons, bitmapOffsets, bitmapTris, numTriangles, numPoints);
		printf("%f\n", costTime);
		//drawFinal(points, tris, nullptr, numPoints + 4, numTriangles, 0);
		int2* cons;
		int3* consTris;
		int numConsTris = 0;
		if (numCons != 0) {
			utils::mallocAndCpy(cons, h_cons, numCons);
			costTime += addConstraints(cubTempStorage, cubTempStorageBytes, points, tris, adjTris, sons, bitmapOffsets, bitmapTris, cons, consTris, numCons, numPoints, numTriangles, numConsTris);
		}
		//drawFinal(points, consTris, cons, numPoints + 4, numConsTris, numCons);
		int3* newAdjTris;
		utils::malloc(newTris, numTriangles);
		utils::malloc(newAdjTris, numTriangles);
		costTime += filterTriangles(cubTempStorage, cubTempStorageBytes, tris, newTris, adjTris, newAdjTris, sons, numTriangles);
		printf("%f\n", costTime);
		if (numConsTris != 0)
		{
			utils::memcpy(newTris + numTriangles, consTris, numConsTris, cudaMemcpyDeviceToDevice);
		}
		numTriangles += numConsTris;
		printf("triangle bf %d\n", numTriangles);
		costTime += delaunayNew(cubTempStorage, cubTempStorageBytes, points, newTris, cons, numPoints + 4, numTriangles, numCons);
		//costTime += delaunay(cubTempStorage, cubTempStorageBytes, points, newTris, numPoints + 4, numTriangles);
		//costTime += reorderPoints(cubTempStorage, cubTempStorageBytes, points, tris, numPoints, numTriangles);
		//tryFastDelaunay(cubTempStorage, cubTempStorageBytes, points, newTris, newAdjTris, numPoints + 4, numTriangles);
		printf("sum time: %f\n", costTime);
		drawFinal(points, newTris, cons, numPoints + 4, numTriangles, numCons);
	}
	else {
		utils::malloc(tris, maxTriangles);
		utils::malloc(adjTris, maxTriangles);
		utils::malloc(sons, maxTriangles);
		rawMesh(cubTempStorage, cubTempStorageBytes, points, tris, adjTris, sons, numTriangles, numPoints);

		int3* newAdjTris;
		utils::malloc(newTris, numTriangles);
		utils::malloc(newAdjTris, numTriangles);
		filterTriangles(cubTempStorage, cubTempStorageBytes, tris, newTris, adjTris, newAdjTris, sons, numTriangles);
		//tryFastDelaunay(cubTempStorage, cubTempStorageBytes, points, newTris, newAdjTris, numPoints + 4, numTriangles);
		//delaunayNew(cubTempStorage, cubTempStorageBytes, points, newTris, numPoints + 4, numTriangles);
		drawFinal(points, newTris, nullptr, numPoints + 4, numTriangles, 0);
	}

	return;
}

