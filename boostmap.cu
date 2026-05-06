#include "math_base.cuh"
#include "utils.cuh"
#include "device_launch_parameters.h"

__global__ void countTriAABB(Point2d* points, int3* tris, int* counts, unsigned int logPointRes, unsigned int logBitmap, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	int3 t = tris[tid];
	Point2d p = points[t.x];
	PointInteger mnx, mny, mxx, mxy;
	mnx = mxx = p.x;
	mny = mxy = p.y;

	p = points[t.y];
	if (p.x > mxx) mxx = p.x; else if (p.x < mnx) mnx = p.x;
	if (p.y > mxy) mxy = p.y; else if (p.y < mny) mny = p.y;

	p = points[t.z];
	if (p.x > mxx) mxx = p.x; else if (p.x < mnx) mnx = p.x;
	if (p.y > mxy) mxy = p.y; else if (p.y < mny) mny = p.y;

	int bmnx = mnx >> (logPointRes - logBitmap);
	int bmxx = mxx >> (logPointRes - logBitmap);
	int bmny = mny >> (logPointRes - logBitmap);
	int bmxy = mxy >> (logPointRes - logBitmap);
	counts[tid] = (bmxx + 1 - bmnx) * (bmxy + 1 - bmny);
}

__device__ bool intersect(const Point2d& a, const Point2d& b, const Point2d* p)
{
	if (a.x < p[0].x && b.x < p[0].x) return false;
	if (a.x > p[2].x && b.x > p[2].x) return false;
	if (a.y < p[0].y && b.y < p[0].y) return false;
	if (a.y > p[2].y && b.y > p[2].y) return false;

	int f = (b.x >= a.x && b.y >= a.y) || (b.x <= a.x && b.y <= a.y);
	return orient2d(a, b, p[f]) * orient2d(a, b, p[f + 2]) <= 0;
}
__device__ bool contains(const Point2d& p, const Point2d& q, const Point2d& a)
{
	return (p.x <= a.x && a.x <= q.x && p.y <= a.y && a.y <= q.y);
}
__device__ bool intersect(const Point2d& a, const Point2d& b, const Point2d& c, const Point2d& x, const Point2d& y)
{
	//if (p.x <= a.x && a.x <= q.x && p.y <= a.y && a.y <= q.y) return true;
	Point2d p[4];
	p[0] = x; p[1] = make_Point2d(y.x, x.y);
	p[2] = y;
	p[3] = make_Point2d(x.x, y.y);

	for (int i = 0; i < 4; i++)
	{
		if (orient2d(a, b, p[i]) >= 0 && orient2d(b, c, p[i]) >= 0 && orient2d(c, a, p[i]) >= 0)
		{
			return true;
		}
	}
	if (intersect(a, b, p)) return true;
	if (intersect(b, c, p)) return true;
	if (intersect(c, a, p)) return true;
	if (contains(x, y, a)) return true;
	if (contains(x, y, b)) return true;
	if (contains(x, y, c)) return true;
	return false;
}
__global__ void intersectSquareTri_s1(Point2d* points, int3* tris, int* offsets, int* counts, int* orders, unsigned int logPointRes, unsigned int logBitmap, int numT, int numI)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numI) return;

	int triId = LBS(offsets, 0, numT, tid);
	int order = tid - offsets[triId];

	int3 t = tris[triId];
	Point2d px = points[t.x];
	Point2d py = points[t.y];
	Point2d pz = points[t.z];

	PointInteger mnx, mny, mxx, mxy;
	mnx = mxx = px.x;
	mny = mxy = px.y;

	if (py.x > mxx) mxx = py.x; else if (py.x < mnx) mnx = py.x;
	if (py.y > mxy) mxy = py.y; else if (py.y < mny) mny = py.y;

	if (pz.x > mxx) mxx = pz.x; else if (pz.x < mnx) mnx = pz.x;
	if (pz.y > mxy) mxy = pz.y; else if (pz.y < mny) mny = pz.y;

	int bmnx = mnx >> (logPointRes - logBitmap);
	int bmxx = mxx >> (logPointRes - logBitmap);
	int bmny = mny >> (logPointRes - logBitmap);
	int bmxy = mxy >> (logPointRes - logBitmap);
	
	int orderx = order / (bmxy - bmny + 1) + bmnx;
	int ordery = (order % (bmxy - bmny + 1)) + bmny;

	PointInteger a = 1;
	a = a << (logPointRes - logBitmap);
	if (intersect(px, py, pz, make_Point2d(a * orderx, a * ordery),
		make_Point2d(a * (orderx + 1) - 1, a * (ordery + 1) - 1)))
	{
		orders[tid] = atomicAdd(&counts[orderx * (1 << logBitmap) + ordery], 1);
	}
	else {
		orders[tid] = -1;
	}
}

__global__ void intersectSquareTri_s2(Point2d* points, int3* tris, int* offsets, int* counts, int* orders, int* ans, unsigned int logPointRes, unsigned int logBitmap, int numT, int numI)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numI) return;

	int triId = LBS(offsets, 0, numT, tid);
	int order = tid - offsets[triId];

	int3 t = tris[triId];
	Point2d px = points[t.x];
	Point2d py = points[t.y];
	Point2d pz = points[t.z];

	PointInteger mnx, mny, mxx, mxy;
	mnx = mxx = px.x;
	mny = mxy = px.y;

	if (py.x > mxx) mxx = py.x; else if (py.x < mnx) mnx = py.x;
	if (py.y > mxy) mxy = py.y; else if (py.y < mny) mny = py.y;

	if (pz.x > mxx) mxx = pz.x; else if (pz.x < mnx) mnx = pz.x;
	if (pz.y > mxy) mxy = pz.y; else if (pz.y < mny) mny = pz.y;

	int bmnx = mnx >> (logPointRes - logBitmap);
	int bmxx = mxx >> (logPointRes - logBitmap);
	int bmny = mny >> (logPointRes - logBitmap);
	int bmxy = mxy >> (logPointRes - logBitmap);

	int orderx = order / (bmxy - bmny + 1) + bmnx;
	int ordery = (order % (bmxy - bmny + 1)) + bmny;

	if (orders[tid] != -1)
	{
		ans[counts[orderx * (1 << logBitmap) + ordery] + orders[tid]] = triId;
	}
}
double getBitmap(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int*& bitmapOffsets, int*& bitmapTris, int numTriangles)
{
	int* countsAABB;
	int* orderInt;
	constexpr int bitmapSize = (1 << logBitmap) * (1<<logBitmap);
	utils::malloc(countsAABB, numTriangles + 1);
	
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::kernel(countTriAABB, numTriangles, 128, points, tris, countsAABB, logPointRes, logBitmap, numTriangles);
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, countsAABB, countsAABB, numTriangles + 1);
	int numInt = utils::getValue(countsAABB, numTriangles);
	
	utils::malloc(bitmapOffsets, bitmapSize + 1);
	utils::malloc(orderInt, numInt);
	utils::kernel(intersectSquareTri_s1, numInt, 128, points, tris, countsAABB, bitmapOffsets, orderInt, logPointRes, logBitmap, numTriangles, numInt);
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, bitmapOffsets, bitmapOffsets, bitmapSize + 1);
	int numAInt = utils::getValue(bitmapOffsets, bitmapSize);
	utils::malloc(bitmapTris, numAInt);
	//printf("%d %d\n", numInt, numAInt);
	utils::kernel(intersectSquareTri_s2, numInt, 128, points, tris, countsAABB, bitmapOffsets, orderInt, bitmapTris, logPointRes, logBitmap, numTriangles, numInt);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "getBitmap" << " time: " << ' ' << elapsedTime << " ms\n";
	
	printf("%d\n", numAInt);
	utils::release(countsAABB, numTriangles + 1);
	utils::release(orderInt, numInt);
	return elapsedTime;
}