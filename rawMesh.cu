#include "math_base.cuh"
#include "utils.cuh"
#include "device_launch_parameters.h"

__global__ void insertBigTriangle(Point2d* points, int* triIds, int3* tris, int3* adjTris, PointInteger res, int numP)
{
	points[numP] = make_Point2d(0, 0);
	points[numP + 1] = make_Point2d(res, 0);
	points[numP + 2] = make_Point2d(res, res);
	points[numP + 3] = make_Point2d(0, res);

	triIds[numP] = -1;
	triIds[numP + 1] = -1;
	triIds[numP + 2] = -1;
	triIds[numP + 3] = -1;
	tris[0] = make_int3(numP, numP + 1, numP + 3);
	tris[1] = make_int3(numP + 1, numP + 2, numP + 3);
	if (adjTris != nullptr) {
		adjTris[0] = make_int3(1, -1, -1);
		adjTris[1] = make_int3(-1, 0, -1);
	}
}

__global__ void insertCornerPoints(Point2d* points, int* triIds, PointInteger res, int numP)
{
	points[numP] = make_Point2d(0, 0);
	points[numP + 1] = make_Point2d(res, 0);
	points[numP + 2] = make_Point2d(res, res);
	points[numP + 3] = make_Point2d(0, res);

	triIds[numP] = -1;
	triIds[numP + 1] = -1;
	triIds[numP + 2] = -1;
	triIds[numP + 3] = -1;
}

__global__ void initTriangle(Point2d* points, int* triIds, int4* onEdge, PointInteger res, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	Point2d RD = make_Point2d(res, 0);
	Point2d LT = make_Point2d(0, res);
	Point2d p = points[tid];
	int o = orient2d(RD, LT, p);
	if (o > 0)
	{
		triIds[tid] = 0;
	}
	else if (o < 0)
	{
		triIds[tid] = 1;
	}
	else {
		triIds[tid] = -2;
		onEdge[tid] = make_int4(0, 1, numP, numP + 2);
	}
}

__global__ void initTriangleBoost(Point2d* points, int3* tris, int* triIds, int4* onEdge, int* bitmapOffset, int* bitmapTris, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	if (triIds[tid] == -1) return;
	//if (tid != 7) return;
	Point2d p = points[tid];

	//printf("%d %d\n", p.x, p.y);

	int orderx, ordery;

	orderx = p.x >> (logPointRes - logBitmap);
	ordery = p.y >> (logPointRes - logBitmap);
	int ad = orderx * (1 << logBitmap) + ordery;
	int L = bitmapOffset[ad];
	int R = bitmapOffset[ad + 1];
	int4 onE;
	onE.x = -1;
	int triId = -3;
	for (int i = L; i < R; i++)
	{
		int ttId = bitmapTris[i];
		int3 t = tris[ttId];
		Point2d x = points[t.x], y = points[t.y], z = points[t.z];
		//printf("%d %d  %d %d  %d %d\n", x.x, x.y, y.x, y.y, z.x, z.y);
		int o1 = orient2d(x, y, p);
		int o2 = orient2d(y, z, p);
		int o3 = orient2d(z, x, p);
		if (o1 > 0 && o2 > 0 && o3 > 0)
		{
			triId = ttId;
			break;
		}
		else if (o1 == 0 && o2 > 0 && o3 > 0)
		{
			triId = -2;
			if (onE.x == -1) {
				onE.x = ttId;
				onE.z = t.z;
			}
			else {
				onE.y = ttId;
				onE.w = t.z;
				break;
			}
		}
		else if (o2 == 0 && o3 > 0 && o1 > 0)
		{
			triId = -2;
			if (onE.x == -1) {
				onE.x = ttId;
				onE.z = t.x;
			}
			else {
				onE.y = ttId;
				onE.w = t.x;
				break;
			}
		}
		else if (o3 == 0 && o1 > 0 && o2 > 0){
			triId = -2;
			if (onE.x == -1) {
				onE.x = ttId;
				onE.z = t.y;
			}
			else {
				onE.y = ttId;
				onE.w = t.y;
				break;
			}
		}
	}
	if (triId == -3)
	{
		printf("error -3\n");
		for (int i = L; i < R; i++)
		{
			int ttId = bitmapTris[i];
			int3 t = tris[ttId];
			Point2d x = points[t.x], y = points[t.y], z = points[t.z];
			//printf("%d %d  %d %d  %d %d\n", x.x, x.y, y.x, y.y, z.x, z.y);
			int o1 = orient2d(x, y, p);
			int o2 = orient2d(y, z, p);
			int o3 = orient2d(z, x, p);
			printf("%d %d %d %d\n", tid, o1, o2, o3);
		}
	}
	triIds[tid] = triId;
 	if (triId == -2)
	{
		onEdge[tid] = onE;
	}
}
__global__ void choosePoint(int* triIds, int4* onEdge, int* cp, int* h_random, int numP)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	//tid = h_random[tid / 32] * 32 + tid % 32;

	int triId = triIds[tid];
	if (triId >= 0) {
		cp[triId] = tid;
	}
	else if (triId == -2)
	{
		int4 f = onEdge[tid];
		if (f.x != -1) {
			//printf("%d try %d\n", tid, f.x);
			cp[f.x] = tid;
		}
		if (f.y != -1) {
			//printf("%d try %d\n", tid, f.y);
			cp[f.y] = tid;
		}
	}
}

__global__ void addNewTriangle_step1(int* cps, int* triIds, int* count, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	int cpId = cps[tid];
	if (cpId == -1)
	{
		count[tid] = 0;
		//printf("%d choos no\n", tid);
	}
	else {
		if (triIds[cpId] == -2)
		{
			count[tid] = 2;
			//printf("%d choos edge %d\n", tid, cpId);
		}
		else {
			count[tid] = 3;
			//printf("%d choos inner %d\n", tid, cpId);
		}
	}
}

__global__ void addNewTriangle_step2(int* cps, int3* tris, int3* adjTris, int* offset, int* sons, int* triIds, int4* onEdge, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	int cpId = cps[tid];
	if (cpId == -1) {
		return;
	}
	int3 ot = tris[tid];
	int sonsOf = numT + offset[tid];
	if (triIds[cpId] == -2) {
		sons[tid] = -sonsOf;
		int4 info = onEdge[cpId];
		int tp;
		if (info.x == tid) {
			tp = info.z;
			onEdge[cpId].x = -1;
		}
		else {
			tp = info.w;
			onEdge[cpId].y = -1;
		}
		int3 t = tris[tid];
		int3 at = adjTris[tid];
		int3 att;

		int3 rt;
		int ax, ay, az;
		if (t.x == tp)
		{
			rt = t;
			az = at.z;
			ay = at.y;
			ax = at.x;
		}
		else if (t.y == tp)
		{
			rt.x = t.y;
			rt.y = t.z;
			rt.z = t.x;
			az = at.x;
			ay = at.z;
			ax = at.y;
		}
		else {
			rt.x = t.z;
			rt.y = t.x;
			rt.z = t.y;
			az = at.y;
			ay = at.x;
			ax = at.z;
		}
		tris[sonsOf] = make_int3(rt.x, rt.y, cpId);
		att.x = ax;
		att.y = sonsOf + 1;
		att.z = az;
		adjTris[sonsOf] = att;

		tris[sonsOf + 1] = make_int3(rt.x, cpId, rt.z);
		att.x = ax;
		att.y = ay;
		att.z = sonsOf;
		adjTris[sonsOf + 1] = att;
	}
	else {
		sons[tid] = sonsOf;
		int3 t = tris[tid];
		int3 at = adjTris[tid];

		tris[sonsOf] = make_int3(cpId, t.y, t.z);
		adjTris[sonsOf] = make_int3(at.x, sonsOf + 1, sonsOf + 2);

		tris[sonsOf + 1] = make_int3(cpId, t.z, t.x);
		adjTris[sonsOf + 1] = make_int3(at.y, sonsOf + 2, sonsOf + 0);

		tris[sonsOf + 2] = make_int3(cpId, t.x, t.y);
		adjTris[sonsOf + 2] = make_int3(at.z, sonsOf + 0, sonsOf + 1);

		triIds[cpId] = -1;
	}
}

__global__ void addNewTriangle_step2_simple(int* cps, int3* tris, int* offset, int* sons, int* triIds, int4* onEdge, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	int cpId = cps[tid];
	if (cpId == -1) {
		return;
	}
	int3 ot = tris[tid];
	int sonsOf = numT + offset[tid];
	if (triIds[cpId] == -2) {
		sons[tid] = -sonsOf;
		int4 info = onEdge[cpId];
		int tp;
		if (info.x == tid) {
			tp = info.z;
			onEdge[cpId].x = -1;
		}
		else {
			tp = info.w;
			onEdge[cpId].y = -1;
		}
		int3 t = tris[tid];
		int3 att;

		int3 rt;
		if (t.x == tp)
		{
			rt = t;
		}
		else if (t.y == tp)
		{
			rt.x = t.y;
			rt.y = t.z;
			rt.z = t.x;
		}
		else {
			rt.x = t.z;
			rt.y = t.x;
			rt.z = t.y;
		}
		tris[sonsOf] = make_int3(rt.x, rt.y, cpId);

		tris[sonsOf + 1] = make_int3(rt.x, cpId, rt.z);
	}
	else {
		sons[tid] = sonsOf;
		int3 t = tris[tid];
		tris[sonsOf] = make_int3(cpId, t.y, t.z);
		tris[sonsOf + 1] = make_int3(cpId, t.z, t.x);
		tris[sonsOf + 2] = make_int3(cpId, t.x, t.y);
		triIds[cpId] = -1;
	}
}


__device__ void scatterPoint(Point2d* points, const Point2d& np, int cpId, int of, const int3& tri, int& newId, int4& onEdge)
{
	Point2d a = points[tri.x];
	Point2d b = points[tri.y];
	Point2d c = points[tri.z];
	Point2d cp = points[cpId];
	int oa = orient2d(a, cp, np);
	int ob = orient2d(b, cp, np);
	int oc = orient2d(c, cp, np);
	bool isDegen = false;
	if (of < 0) {
		of = -of;
		isDegen = true;
	}
	newId = -2;

	if (!isDegen)
	{
		if (oa < 0 && ob > 0)
		{
			newId = of + 2;
		}
		else if (ob < 0 && oc > 0)
		{
			newId = of;
		}
		else if (oc < 0 && oa > 0)
		{
			newId = of + 1;
		}
		else {
			int tag = 0;
			if (oa == 0) tag = 1;
			if (ob == 0) tag = 2;

			onEdge = make_int4(of + (tag % 3), of + ((tag + 1) % 3),
				getInt3(&tri, (tag + 1) % 3), getInt3(&tri, tag % 3));
		}
	}
	else if (oa != 0 && ob != 0 && oc != 0) {
		int cnt = 0;
		if (oa > 0) cnt++;
		if (ob > 0) cnt++;
		if (oc > 0) cnt++;
		newId = of + cnt - 1;
	}
	else {
		if (oa == 0 && ob != 0 && oc != 0)
		{
			onEdge = make_int4(of, of + 1, tri.y, tri.z);
		}
		if (ob == 0 && oa != 0 && oc != 0)
		{
			onEdge = make_int4(of, of + 1, tri.z, tri.x);
		}
		if (oc == 0 && oa != 0 && ob != 0)
		{
			onEdge = make_int4(of, of + 1, tri.x, tri.y);
		}
		if (oa != 0 && ob == 0 && oc == 0)
		{
			onEdge.x = tri.x;
			newId = of + (oa > 0);
		}
		if (ob != 0 && oa == 0 && oc == 0)
		{
			onEdge.x = tri.y;
			newId = of + (ob > 0);
		}
		if (oc != 0 && oa == 0 && ob == 0)
		{
			onEdge.x = tri.z;
			newId = of + (oc > 0);
		}
	}
}


__global__ void scatterPoints(Point2d* points, int* triIds, int3* tris, int* cps, int* sonsOf, int4* onEdge, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	int triId = triIds[tid];
	if (triId == -1) return;

	Point2d np = points[tid];
	int newId;
	int4 newOnEdge;

	if (triId != -2)
	{
		int cpId = cps[triId];
		scatterPoint(points, np, cpId, sonsOf[triId], tris[triId], newId, newOnEdge);
		triIds[tid] = newId;
		if (newId == -2) {
			onEdge[tid] = newOnEdge;
		}
	}
	else {
		int4 info = onEdge[tid];
		if (info.x != -1)
		{
			int cpId = cps[info.x];
			scatterPoint(points, np, cpId, sonsOf[info.x], tris[info.x], newId, newOnEdge);
			int3 t = tris[newId];
			if (!contains(t, info.z))
			{
				onEdge[tid].z = cpId;
			}
			onEdge[tid].x = newId;
		}
		if (info.y != -1)
		{
			int cpId = cps[info.y];
			scatterPoint(points, np, cpId, sonsOf[info.y], tris[info.y], newId, newOnEdge);
			int3 t = tris[newId];
			if (!contains(t, info.w))
			{
				onEdge[tid].w = cpId;
			}
			onEdge[tid].y = newId;
		}
	}

	//printf("state %d %d\n", tid, triIds[tid]);
}

__global__ void scatterPoints_f(Point2d* points, int* triIds, int3* tris, int* cps, int* sonsOf, int4* onEdge, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	int triId = triIds[tid];
	if (triId == -1) return;

	Point2d np = points[tid];
	int newId;
	int4 newOnEdge;

	if (triId != -2)
	{
		int cpId = cps[triId];
		scatterPoint(points, np, cpId, sonsOf[triId], tris[triId], newId, newOnEdge);
		triIds[tid] = newId;
		if (newId == -2) {
			onEdge[tid] = newOnEdge;
		}
	}
	else {
		int4 info = onEdge[tid];
		if (info.x != -1)
		{
			int cpId = cps[info.x];
			scatterPoint(points, np, cpId, sonsOf[info.x], tris[info.x], newId, newOnEdge);
			int3 t = tris[newId];
			if (!contains(t, info.z))
			{
				onEdge[tid].z = cpId;
			}
			onEdge[tid].x = newId;
		}
		if (info.y != -1)
		{
			int cpId = cps[info.y];
			scatterPoint(points, np, cpId, sonsOf[info.y], tris[info.y], newId, newOnEdge);
			int3 t = tris[newId];
			if (!contains(t, info.w))
			{
				onEdge[tid].w = cpId;
			}
			onEdge[tid].y = newId;
		}
	}

	//printf("state %d %d\n", tid, triIds[tid]);
}


__device__ void getSonTris(Point2d* points, int3* tris, int& triId, int& edgeId, int* sons, int sonId, const int x, const int y)
{
	if (sonId > 0)
	{
		triId = sonId + edgeId;
		edgeId = 0;
	}
	else {
		int3 t = tris[triId];
		int3 tson1 = tris[-sonId];
		if (t.x == tson1.x)
		{
			if (edgeId == 0)
			{
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (inner(p1, p2, points[x]) && inner(p1, p2, points[y]))
					triId = -sonId;
				else
					triId = -sonId + 1;
				edgeId = 0;
			}
			else if (edgeId == 1) {
				triId = -sonId + 1;
				edgeId = 1;
			}
			else { // == 2
				triId = -sonId;
				edgeId = 2;
			}
		}
		else if (t.y == tson1.x)
		{
			if (edgeId == 0)
			{
				triId = -sonId;
				edgeId = 2;
			}
			else if (edgeId == 1)
			{
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (inner(p1, p2, points[x]) && inner(p1, p2, points[y]))
					triId = -sonId;
				else
					triId = -sonId + 1;
				edgeId = 0;
			}
			else {
				triId = -sonId + 1;
				edgeId = 1;
			}
		}
		else {
			if (edgeId == 0)
			{
				triId = -sonId + 1;
				edgeId = 1;

			}
			else if (edgeId == 1)
			{
				triId = -sonId;
				edgeId = 2;
			}
			else {
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (inner(p1, p2, points[x]) && inner(p1, p2, points[y]))
					triId = -sonId;
				else
					triId = -sonId + 1;
				edgeId = 0;
			}
		}
	}
}

__global__ void updateAdjTris_new(Point2d* points, int3* tris, int3* adjTris, int* sons, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	if (sons[tid] != -1) return;
	int3 at = adjTris[tid];
	int3 t = tris[tid];

	int triId = at.x;
	int sonId = -1;
	int edgeId;
	if (triId != -1) {
		sonId = sons[triId];
		int3 o = tris[triId];
		Point2d a = points[t.z], b = points[t.y];
		if (orient2d(a, b, points[o.x]) != 0)
		{
			edgeId = 0;
		}
		else if (orient2d(a, b, points[o.y]) != 0)
		{
			edgeId = 1;
		}
		else {
			edgeId = 2;
		}
	}
	Point2d a = points[t.z], b = points[t.y];
	while (sonId != -1)
	{
		getSonTris(points, tris, triId, edgeId, sons, sonId, t.y, t.z);
		sonId = sons[triId];
	}
	at.x = triId;

	triId = at.y;
	sonId = -1;
	if (triId != -1) {
		sonId = sons[triId];
		int3 o = tris[triId];
		Point2d a = points[t.z], b = points[t.x];
		if (orient2d(a, b, points[o.x]) != 0)
		{
			edgeId = 0;
		}
		else if (orient2d(a, b, points[o.y]) != 0)
		{
			edgeId = 1;
		}
		else {
			edgeId = 2;
		}
	}
	while (sonId != -1)
	{
		getSonTris(points, tris, triId, edgeId, sons, sonId, t.x, t.z);
		sonId = sons[triId];
	}
	at.y = triId;

	triId = at.z;
	sonId = -1;
	if (triId != -1) {
		sonId = sons[triId];
		int3 o = tris[triId];
		Point2d a = points[t.x], b = points[t.y];
		if (orient2d(a, b, points[o.x]) != 0)
		{
			edgeId = 0;
		}
		else if (orient2d(a, b, points[o.y]) != 0)
		{
			edgeId = 1;
		}
		else {
			edgeId = 2;
		}
	}
	while (sonId != -1)
	{
		getSonTris(points, tris, triId, edgeId, sons, sonId, t.x, t.y);
		sonId = sons[triId];
	}
	at.z = triId;
	//printf("%d %d %d %d\n", tid, at.x, at.y, at.z);
	adjTris[tid] = at;
}

__global__ void samplePoints_s1(Point2d* points, int* sample, int* counts, unsigned int logRes, unsigned int logSampleSize, int numP)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;

	Point2d p = points[tid];
	int x = (p.x >> (logRes - logSampleSize));
	int y = (p.y >> (logRes - logSampleSize));
	int ad = x * (1 << logSampleSize) + y;
	sample[ad] = tid;
	counts[ad] = 1;
}

__global__ void samplePoints_s2(int* sample, int* newSample, int* offsets, int numS)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numS) return;

	int t = sample[tid];
	if (t != -1)
	{
		newSample[offsets[tid]] = t;
	}
}
__global__ void samplePoints_s3(Point2d* points, Point2d* samplePoints, int* sample, int numS)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numS) return;
	samplePoints[tid] = points[sample[tid]];
}


#include <random>
#include <algorithm>

double getSamplePoints(void* cubTemp, size_t cubSize, Point2d* points, int numPoints, Point2d*& samplePoints, int*& sample, int& numSample)
{
	numSample = (1 << logSample) * (1 << logSample);
	utils::malloc(sample, numSample + 4);
	utils::malloc(samplePoints, numSample + 4);

	int* sampleTemp;
	utils::malloc(sampleTemp, numSample);
	int* sampleCounts;
	utils::malloc(sampleCounts, numSample + 1);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::memset(sampleCounts, numSample + 1, 0);
	utils::memset(sampleTemp, numSample, -1);
	utils::kernel(samplePoints_s1, numPoints, 128, points, sampleTemp, sampleCounts, logPointRes, logSample, numPoints);
	utils::exlusiveScan(cubTemp, cubSize, sampleCounts, sampleCounts, numSample + 1);
	utils::kernel(samplePoints_s2, numSample, 128, sampleTemp, sample, sampleCounts, numSample);
	
	numSample = utils::getValue(sampleCounts, numSample);
	utils::kernel(samplePoints_s3, numSample, 128, points, samplePoints, sample, numSample);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "sample points " << numSample <<" time: "<< elapsedTime << " ms\n";
	utils::release(sampleCounts, numSample + 1);
	utils::release(sampleTemp, numSample);
	return elapsedTime;
}

__global__ void choosePointSmart_s1(Point2d* points, int3* tris, int* triIds, int4* onEdge, int* cp, int2* weights, int numP)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	//tid = h_random[tid / 32] * 32 + tid % 32;

	int triId = triIds[tid];
	if (triId > 0) {
		int3 t = tris[triId];
		Point2d a = points[t.x], b = points[t.y], c = points[t.z];
		Point2d center = points[tid];
		PointInteger temp = (a.x + b.x + c.x) - center.x * 3;
		if (temp < 0) temp = -temp;
		PointInteger temp2 = (a.y + b.y + c.y) - center.y * 3;
		if (temp2 < 0) temp2 = -temp2;
		temp = temp + temp2;

		temp = temp - (1 << (logPointRes + 1));
		weights[tid].x = temp;
		///atomicMin(&cp[triId], -tid-2);
		//cp[triId] = tid;
		atomicMax(&cp[triId], tid);
	}
	//else if (triId == -2) {
	//	int4 f = onEdge[tid];
	//	if (f.x != -1) {
	//		int3 t = tris[f.x];
	//		Point2d a = points[t.x], b = points[t.y], c = points[t.z];
	//		Point2d center = points[tid];
	//		PointInteger temp = (a.x + b.x + c.x) - center.x * 3;
	//		if (temp.isNegative()) temp = -temp;
	//		PointInteger temp2 = (a.y + b.y + c.y) - center.y * 3;
	//		if (temp2.isNegative()) temp2 = -temp2;
	//		temp = temp + temp2;
	//
	//		temp = temp - (1 << (logPointRes + 1));
	//		weights[tid].x = temp;
	//		atomicMin(&cp[f.x], temp);
	//	}
	//	if (f.y != -1) {
	//		int3 t = tris[f.y];
	//		Point2d a = points[t.x], b = points[t.y], c = points[t.z];
	//		Point2d center = points[tid];
	//		PointInteger temp = (a.x + b.x + c.x) - center.x * 3;
	//		if (temp.isNegative()) temp = -temp;
	//		PointInteger temp2 = (a.y + b.y + c.y) - center.y * 3;
	//		if (temp2.isNegative()) temp2 = -temp2;
	//		temp = temp + temp2;
	//
	//		temp = temp - (1 << (logPointRes + 1));
	//		weights[tid].y = temp;
	//		atomicMin(&cp[f.y], temp);
	//	}
	//}
}

__global__ void choosePointSmart_s2(int* triIds, int4* onEdge, int* cp, int2* weights, int numP)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;
	//tid = h_random[tid / 32] * 32 + tid % 32;
	int triId = triIds[tid];
	//if (triId > 0) {
	//	if (-tid-2 == cp[triId]) {
	//		cp[triId] = tid;
	//	}
	//}
	//else if (triId == -2)
	//{
	//	int4 f = onEdge[tid];
	//	if (f.x != -1)
	//	{
	//		if (weights[tid].x == cp[f.x])
	//		{
	//			cp[f.x] = tid;
	//		}
	//	}
	//	if (f.y != -1)
	//	{
	//		if (weights[tid].y == cp[f.y])
	//		{
	//			cp[f.y] = tid;
	//		}
	//	}
	//}
}

__global__ void sortPoints_s1(int* triIds, int4* onEdge, int* counts, int* orders, int slotSize, int numP)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;

	int triId = triIds[tid];
	if (triId == -1) return;
	if (triId == -2)
	{
		triId = onEdge[tid].x;
	}

	orders[tid] = atomicAdd(counts + triId * slotSize + tid % slotSize, 1);
}

__global__ void sortPoints_s2(Point2d* points, Point2d* newPoints, int* triIds, int4* onEdge, int* counts, int* orders, int* newOrder, int slotSize, int numP)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numP) return;

	int triId = triIds[tid];
	if (triId == -1) return;
	
	if (triId == -2)
	{
		triId = onEdge[tid].x;
	}

	int offset = counts[triId * slotSize + tid % slotSize] + orders[tid];
	newOrder[tid] = offset;
	newPoints[offset] = points[tid];
}
void rawMeshInsert(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int* triIds, int4* onEdges, int& numTriangles, int numPoints)
{
	const int countSize = 1024 * 100;
	const int maxTriangles = numPoints * 4;
	int* chosenPoints;
	int* countNewTriangles;
	int2* weights;
	int* counts;
	int* orders;
	utils::malloc(counts, countSize);
	utils::malloc(orders, numPoints);
	utils::malloc(weights, numPoints);
	utils::malloc(chosenPoints, maxTriangles);
	utils::malloc(countNewTriangles, maxTriangles);
	int* d_random;
	int step1 = 0;
	while (true) {
		step1++;
		//utils::memset(chosenPoints, numTriangles);
		utils::memset(chosenPoints, numTriangles, -1);
		//utils::kernel(choosePointSmart_s1, numPoints, 128, points, tris, triIds, onEdges, chosenPoints, weights, numPoints);
		//utils::kernel(choosePointSmart_s2, numPoints, 128, triIds, onEdges, chosenPoints, weights, numPoints);
		utils::kernel(choosePoint, numPoints, 128, triIds, onEdges, chosenPoints, d_random, numPoints);
		utils::kernel(addNewTriangle_step1, numTriangles, 128, chosenPoints, triIds, countNewTriangles, numTriangles);
		utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, countNewTriangles, countNewTriangles, numTriangles + 1);
		int newc = utils::getValue(countNewTriangles, numTriangles);
		if (newc == 0) { printf("exit %d\n", step1);  break; }
		utils::kernel(addNewTriangle_step2, numTriangles, 128, chosenPoints, tris, adjTris, countNewTriangles, sons, triIds, onEdges, numTriangles);
		//utils::kernel(addNewTriangle, numTriangles, 128, chosenPoints, tris, int* offset, int* sons, int* triIds, int numT)
		utils::kernel(scatterPoints, numPoints, 128, points, triIds, tris, chosenPoints, sons, onEdges, numPoints);
		
		numTriangles += newc;
		//int slotSize = (countSize / numTriangles) > 1 ? (countSize / numTriangles) : 1;
		//utils::kernel(sortPoints_s1, numPoints, 128, triIds, onEdges, counts, orders, slotSize, numPoints);
		//sortPoints_s2(points, newPoints, triIds, onEdges, counts, orders, newOrder, slotSize, numPoints);
	}
	utils::release(weights, numPoints);
	utils::release(countNewTriangles, maxTriangles);
	utils::release(chosenPoints, maxTriangles);
	utils::release(counts, countSize);
	utils::release(orders, numPoints);
}

__global__ void excludeSamples(int3* tris, int* triIds, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;

	int3 t = tris[tid];
	triIds[t.x] = -1;
	triIds[t.y] = -1;
	triIds[t.z] = -1;
}


__device__ bool testAdj(const Point2d& x, const Point2d& y, const Point2d& l, const Point2d& r)
{
	if (orient2d(x, y, l) != 0) return false;
	if (orient2d(x, y, r) != 0) return false;
	if (!inner(l, r, x)) return false;
	if (!inner(l, r, y)) return false;
	return true;
}

__global__ void testAdjTris(int3* tris, int3* adjTris, int* sons, int numT)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numT) return;
	if (sons[tid] != -1) return;
	int3 at = adjTris[tid];
	int3 t = tris[tid];

	if (at.x != -1) {
		int3 tx = tris[at.x];
		if (sons[at.x] != -1) { printf("not son x\n"); }
		if (!contains(tx, t.y))
		{
			printf("%d x c y\n", tid);
		}
		if (!contains(tx, t.z))
		{
			printf("%d x c z\n", tid);
		}
		if (!contains(adjTris[at.x], tid))
		{
			printf("%d x not contains\n", tid);
		}
	}
	else printf("edge!\n");

	if (at.y != -1) {
		int3 ty = tris[at.y];
		if (sons[at.y] != -1) { printf("not son y\n"); }
		if (!contains(ty, t.x))
		{
			printf("%d y c x\n", tid);
		}
		if (!contains(ty, t.z))
		{
			printf("%d y c z\n", tid);
		}
		if (!contains(adjTris[at.y], tid))
		{
			printf("%d y not contains\n", tid);
		}
	}
	else printf("edge!\n");

	if (at.z != -1) {
		int3 tz = tris[at.z];
		if (sons[at.z] != -1) { printf("not son z\n"); }
		if (!contains(tz, t.x))
		{
			printf("%d z c x\n", tid);
		}
		if (!contains(tz, t.y))
		{
			printf("%d z c y\n", tid);
		}
		if (!contains(adjTris[at.z], tid))
		{
			printf("%d z not contains\n", tid);
		}
	}
	else printf("edge!\n");
}

double rawMeshAfter(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int* bitmapOffsets, int* bitmapTriIds, int& numTriangles, int numPoints)
{
	MyInteger<logPointRes> res;
	//int res;
	{
		long long tempRes = 1LL << logPointRes;
		res = tempRes - 1;
	}
	const int maxTriangles = numPoints * 4;
	int* triIds;
	utils::malloc(triIds, numPoints + 4);
	int4* onEdge;
	utils::malloc(onEdge, numPoints + 4);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float c1, c2, c3, c4;
	c1 = c2 = c3 = c4 = 0;
	utils::memset(triIds, numPoints + 4);
	utils::memset(onEdge, numPoints + 4);
	utils::memset(sons, maxTriangles, -1);
	utils::kernel(insertCornerPoints, 1, 1, points, triIds, res, numPoints);
	
	utils::kernel(excludeSamples, numTriangles, 128, tris, triIds, numTriangles);

	utils::kernel(initTriangleBoost, numPoints, 128, points, tris, triIds, onEdge, bitmapOffsets, bitmapTriIds, numPoints);
	
	rawMeshInsert(cubTempStorage, cubTempStorageBytes, points, tris, adjTris, sons, triIds, onEdge, numTriangles, numPoints);

	utils::kernel(updateAdjTris_new, numTriangles, 128, points, tris, adjTris, sons, numTriangles);
	//utils::kernel("updateAdj", updateAdjTris, numTriangles, 128, tris, adjTris, sons, numTriangles);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "newTriangles" << " time: " << ' ' << elapsedTime << " ms\n";
	printf("%f %f %f %f\n", c1, c2, c3, c4);
	//utils::kernel(testAdjTris, numTriangles, 128, tris, adjTris, sons, numTriangles);

	utils::release(triIds, numPoints + 4);
	utils::release(onEdge, numPoints + 4);
	return elapsedTime;
}

double rawMeshBefore(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int* sons, int& numTriangles, int numPoints)
{
	MyInteger<logPointRes> res;
	//int res;
	{
		MyInteger<logPointRes> tempRes = 1LL << logPointRes;
		res = tempRes - 1;
	}
	const int maxTriangles = numPoints * 4;
	int* triIds;
	utils::malloc(triIds, numPoints + 4);
	int4* onEdge;
	utils::malloc(onEdge, numPoints + 4);

	numTriangles = 2;
	int* chosenPoints;
	int* countNewTriangles;
	utils::malloc(chosenPoints, maxTriangles);
	utils::malloc(countNewTriangles, maxTriangles);


	//int numWarps = (numPoints + 31) / 32;
	//int* random = new int[numWarps];
	//for (int i = 0; i < numWarps; i++)
	//{
	//	random[i] = i;
	//}
	//std::mt19937 gen(199);
	//std::shuffle(random, random + numWarps, gen);
	int* d_random;
	//utils::mallocAndCpy(d_random, random, numWarps);
	//delete[] random;

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float c1, c2, c3, c4;
	c1 = c2 = c3 = c4 = 0;
	utils::memset(onEdge, numPoints + 4);
	utils::memset(sons, maxTriangles, -1);

	utils::kernel(insertBigTriangle, 1, 1, points, triIds, tris, (int3*)nullptr, res, numPoints);
	utils::kernel(initTriangle, numPoints, 128, points, triIds, onEdge, res, numPoints);
	int step1 = 0;
	while (true) {

		step1++;
		utils::memset(chosenPoints, numTriangles, -1);
		utils::kernel(choosePoint, numPoints, 128, triIds, onEdge, chosenPoints, d_random, numPoints);
		utils::kernel(addNewTriangle_step1, numTriangles, 128, chosenPoints, triIds, countNewTriangles, numTriangles);
		utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, countNewTriangles, countNewTriangles, numTriangles + 1);
		int newc = utils::getValue(countNewTriangles, numTriangles);
		if (newc == 0) { break; }
		utils::kernel(addNewTriangle_step2_simple, numTriangles, 128, chosenPoints, tris, countNewTriangles, sons, triIds, onEdge, numTriangles);
		//utils::kernel(addNewTriangle, numTriangles, 128, chosenPoints, tris, int* offset, int* sons, int* triIds, int numT)
		utils::kernel(scatterPoints, numPoints, 128, points, triIds, tris, chosenPoints, sons, onEdge, numPoints);
		numTriangles += newc;
		//printf("%d %d\n", newc, numTriangles);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "newTriangles" << " time: " << step1 << ' ' << elapsedTime << " ms\n";
	//utils::kernel(testAdjTris, numTriangles, 128, tris, adjTris, sons, numTriangles);

	utils::release(triIds, numPoints + 4);
	utils::release(onEdge, numPoints + 4);
	utils::release(chosenPoints, maxTriangles);
	utils::release(countNewTriangles, maxTriangles);
	//utils::release(d_random, numWarps);
	return elapsedTime;
}
void rawMesh(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int& numTriangles, int numPoints)
{
	MyInteger<logPointRes> res;
	//int res;
	{
		long long tempRes = 1LL << logPointRes;
		res = tempRes - 1;
	}
	const int maxTriangles = numPoints * 4;
	int* triIds;
	utils::malloc(triIds, numPoints + 4);
	int4* onEdge;
	utils::malloc(onEdge, numPoints + 4);
	
	numTriangles = 2;
	int* chosenPoints;
	int* countNewTriangles;
	utils::malloc(chosenPoints, maxTriangles);
	utils::malloc(countNewTriangles, maxTriangles);
	
	
	//int numWarps = (numPoints + 31) / 32;
	//int* random = new int[numWarps];
	//for (int i = 0; i < numWarps; i++)
	//{
	//	random[i] = i;
	//}
	//std::mt19937 gen(199);
	//std::shuffle(random, random + numWarps, gen);
	int* d_random;
	//utils::mallocAndCpy(d_random, random, numWarps);
	//delete[] random;

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float c1, c2, c3, c4;
	c1 = c2 = c3 = c4 = 0;
	utils::memset(onEdge, numPoints + 4);
	utils::memset(sons, maxTriangles, -1);

	utils::kernel(insertBigTriangle, 1, 1, points, triIds, tris, adjTris, res, numPoints);
	utils::kernel(initTriangle, numPoints, 128, points, triIds, onEdge, res, numPoints);
	int step1 = 0;
	while (true) {
		
		step1++;
		utils::memset(chosenPoints, numTriangles, -1);
		utils::kernel(choosePoint, numPoints, 128, triIds, onEdge, chosenPoints, d_random, numPoints);
		utils::kernel(addNewTriangle_step1, numTriangles, 128, chosenPoints, triIds, countNewTriangles, numTriangles);
		utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, countNewTriangles, countNewTriangles, numTriangles + 1);
		int newc = utils::getValue(countNewTriangles, numTriangles);
		if (newc == 0) { printf("exit %d\n", step1);  break; }
		utils::kernel(addNewTriangle_step2, numTriangles, 128, chosenPoints, tris, adjTris, countNewTriangles, sons, triIds, onEdge, numTriangles);
		//utils::kernel(addNewTriangle, numTriangles, 128, chosenPoints, tris, int* offset, int* sons, int* triIds, int numT)
		utils::kernel(scatterPoints, numPoints, 128, points, triIds, tris, chosenPoints, sons, onEdge, numPoints);
		numTriangles += newc;
		//printf("%d %d\n", newc, numTriangles);
		
	}
	utils::kernel(updateAdjTris_new, numTriangles, 128, points, tris, adjTris, sons, numTriangles);
	//utils::kernel("updateAdj", updateAdjTris, numTriangles, 128, tris, adjTris, sons, numTriangles);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << "newTriangles" << " time: " << step1 << ' ' << elapsedTime << " ms\n";
	printf("%f %f %f %f\n", c1, c2, c3, c4);
	//utils::kernel(testAdjTris, numTriangles, 128, tris, adjTris, sons, numTriangles);

	utils::release(triIds, numPoints + 4);
	utils::release(onEdge, numPoints + 4);
	utils::release(chosenPoints, maxTriangles);
	utils::release(countNewTriangles, maxTriangles);
	//utils::release(d_random, numWarps);
}