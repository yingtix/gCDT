#include "math_base.cuh"
#include "utils.cuh"

void drawOverlaps(Point2d* points, int3* tris, int* triIds, int numOverlaps, int numPoints, int numTriangles)
{
}

void drawPolygons(Point2d* points, int* polyP, int* polyC, int numPoints, int numPolyP)
{
}
__global__ void locatePointsLB_s1(Point2d* points, int2* cons, unsigned long long* counts, unsigned long long* sum, int logRes, int numC)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numC) return;
	Point2d p1 = points[cons[tid].x];
	Point2d p2 = points[cons[tid].y];
	PointInteger w1 = (p1.x - p2.x);
	if (w1 < 0) w1 = -w1;
	PointInteger w2 = (p1.y - p2.y);
	if (w2 < 0) w1 -= w2; else w1 += w2;
	if (logRes > 32) w1 >>= (logRes - 32);
	counts[tid] = w1;
	atomicAdd(sum, (unsigned long long)w1);
}

__global__ void locatePointsLB_s2(unsigned long long* counts, int* segs, unsigned long long* sum, int maxT, int numC)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numC) return;
	unsigned long long wl = counts[tid];
	int tt = 1.0 * wl / (*sum) * maxT;
	if (tt >= maxT) tt = maxT - 1;
	segs[tid] = tt < 1 ? 1 : tt;
}

__device__ HomoPoint2d make_homoPoint(const Point2d& p1, const Point2d& p2, int i, int x)
{
	HomoPoint2d r;
	HomoPointInteger t = x;
	t *= p1.x;
	r.x = i;
	r.x *= (p2.x - p1.x);
	r.x += t;
	
	t = x;
	t *= p1.y;
	r.y = i;
	r.y *= (p2.y - p1.y);
	r.y += t;
	r.z = x;
	return r;
}
__device__ HomoPoint2d make_homoPoint(const Point2d& p)
{
	HomoPoint2d ret;
	ret.x = p.x;
	ret.y = p.y;
	ret.z = 1;
	return ret;
}

//1 2 3: son triX   son triY   son triZ
//-1 -2 -3: edge[YZ]  edge[XZ]  edge[XY]
__device__ int intSegmentTriangleFirst(const Point2d& s1, const Point2d& s2, const HomoPoint2d& h1,
	const Point2d& t1, const Point2d& t2, const Point2d& t3, const Point2d& cp)
{
	//printf("	%d %d\n", (int)t1.x, (int)t1.y);
	//printf("	%d %d\n", (int)t2.x, (int)t2.y);
	//printf("	%d %d\n", (int)t3.x, (int)t3.y);
	int oc = orient2d(s1, s2, cp);
	int o1c1 = orient2d(t1, cp, h1);
	int o2c1 = orient2d(t2, cp, h1);
	int o3c1 = orient2d(t3, cp, h1);

	if ((o1c1 < 0 && o2c1 > 0) || (o1c1 == 0 && o2c1 > 0 && oc > 0) || (o1c1 < 0 && o2c1 == 0 && oc < 0)) {
		return 3;
	}

	if ((o2c1 < 0 && o3c1 > 0) || (o2c1 == 0 && o3c1 > 0 && oc > 0) || (o2c1 < 0 && o3c1 == 0 && oc < 0)) {
		return 1;
	}

	if ((o3c1 < 0 && o1c1 > 0) || (o3c1 == 0 && o1c1 > 0 && oc > 0) || (o3c1 < 0 && o1c1 == 0 && oc < 0))
	{
		return 2;
	}

	//oc == 0
	int o1c2 = orient2d(t1, cp, s2);
	int o2c2 = orient2d(t2, cp, s2);
	int o3c2 = orient2d(t3, cp, s2);
	if (h1 == cp) {
		if (o1c2 < 0 && o2c2 > 0)
		{
			return 3;
		}
		if (o2c2 < 0 && o3c2 > 0)
		{
			return 1;
		}
		if (o3c2 < 0 && o1c2 > 0)
		{
			return 2;
		}
	}
	if (o1c1 == 0 && o1c2 == 0) return -1;
	if (o2c1 == 0 && o2c2 == 0) return -2;
	//if (o3c1 == 0 && o3c2 == 0) return -3;
	//debug
	return -3;
}

//1 2 3: son triX   son triY   son triZ
__device__ int intSegmentTriangleFirstSimple(const Point2d& s1, const Point2d& s2, const HomoPoint2d& h1,
	const Point2d& t1, const Point2d& t2, const Point2d& t3, const Point2d& cp)
{
	int oc = orient2d(s1, s2, cp);
	int o1c1 = orient2d(t1, cp, h1);
	int o2c1 = orient2d(t2, cp, h1);
	int o3c1 = orient2d(t3, cp, h1);

	if ((o1c1 < 0 && o2c1 > 0) || (o1c1 == 0 && o2c1 > 0 && oc > 0) || (o1c1 < 0 && o2c1 == 0 && oc < 0)) {
		return 3;
	}

	if ((o2c1 < 0 && o3c1 > 0) || (o2c1 == 0 && o3c1 > 0 && oc > 0) || (o2c1 < 0 && o3c1 == 0 && oc < 0)) {
		return 1;
	}

	if ((o3c1 < 0 && o1c1 > 0) || (o3c1 == 0 && o1c1 > 0 && oc > 0) || (o3c1 < 0 && o1c1 == 0 && oc < 0))
	{
		return 2;
	}

	//oc == 0
	int o1c2 = orient2d(t1, cp, s2);
	int o2c2 = orient2d(t2, cp, s2);
	int o3c2 = orient2d(t3, cp, s2);
	if (h1 == cp) {
		if (o1c2 < 0 && o2c2 > 0)
		{
			return 3;
		}
		if (o2c2 < 0 && o3c2 > 0)
		{
			return 1;
		}
		if (o3c2 < 0 && o1c2 > 0)
		{
			return 2;
		}
	}
	if (o1c1 + o1c2 >= 1) return 2;
	if (o1c2 + o1c2 <= -1) return 3;
	if (o2c1 + o2c2 >= 1) return 3;
	if (o2c1 + o2c2 <= -1) return 1;
	if (o3c1 + o3c2 >= 1) return 1;
	return 2;
}

__device__ int intSegmentTriangleFirst(const Point2d& s1, const Point2d& s2, const Point2d& h1,
	const Point2d& t1, const Point2d& t2, const Point2d& t3, const Point2d& cp)
{
	//printf("	%d %d\n", (int)t1.x, (int)t1.y);
	//printf("	%d %d\n", (int)t2.x, (int)t2.y);
	//printf("	%d %d\n", (int)t3.x, (int)t3.y);
	//{
	//	if (orient2d(t1, t2, h1) < 0 || orient2d(t2, t3, h1) < 0 || orient2d(t3, t1, h1) < 0)
	//	{
	//		printf("not In!\n");
	//	}
	//}
	int oc = orient2d(s1, s2, cp);
	int o1c1 = orient2d(t1, cp, h1);
	int o2c1 = orient2d(t2, cp, h1);
	int o3c1 = orient2d(t3, cp, h1);

	if ((o1c1 < 0 && o2c1 > 0) || (o1c1 == 0 && o2c1 > 0 && oc > 0) || (o1c1 < 0 && o2c1 == 0 && oc < 0)) {
		return 3;
	}

	if ((o2c1 < 0 && o3c1 > 0) || (o2c1 == 0 && o3c1 > 0 && oc > 0) || (o2c1 < 0 && o3c1 == 0 && oc < 0)) {
		return 1;
	}

	if ((o3c1 < 0 && o1c1 > 0) || (o3c1 == 0 && o1c1 > 0 && oc > 0) || (o3c1 < 0 && o1c1 == 0 && oc < 0))
	{
		return 2;
	}

	//oc == 0
	int o1c2 = orient2d(t1, cp, s2);
	int o2c2 = orient2d(t2, cp, s2);
	int o3c2 = orient2d(t3, cp, s2);
	if (h1 == cp) {
		if (o1c2 < 0 && o2c2 > 0)
		{
			return 3;
		}
		if (o2c2 < 0 && o3c2 > 0)
		{
			return 1;
		}
		if (o3c2 < 0 && o1c2 > 0)
		{
			return 2;
		}
	}
	if (o1c1 == 0 && o1c2 == 0) return -1;
	if (o2c1 == 0 && o2c2 == 0) return -2;
	//if (o3c1 == 0 && o3c2 == 0) return -3;
	//debug
	return -3;

}
__device__ int locateSegmentOnEdge(int3* tris, Point2d* points, const HomoPoint2d& realStart, const Point2d& s2, int* sons, int& t1, int& e1, int& t2, int& e2)
{
	int sonId = sons[t1];
	while (sonId != -1)
	{
		if (sonId > 0)
		{
			t1 = sonId + e1;
			e1 = 0;
		}
		else {
			int3 t = tris[t1];
			int3 tson1 = tris[-sonId];
			int tag = 2;
			if (t.x == tson1.x) tag = 0;
			if (t.y == tson1.x) tag = 1;

			if (e1 == tag)
			{
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (realStart == p2)
				{
					if (inner(p2, s2, p1) || inner(p2, p1, s2))
					{
						t1 = -sonId;
					}
					else {
						t1 = -sonId + 1;
					}
				}
				else {
					if (inner(p1, p2, realStart))
					{
						t1 = -sonId;
					}
					else {
						t1 = -sonId + 1;
					}
				}
				e1 = 0;
			}
			else if (e1 == (tag + 1) % 3)
			{
				t1 = -sonId + 1;
				e1 = 1;
			}
			else {
				t1 = -sonId;
				e1 = 2;
			}
		}
		sonId = sons[t1];
	}

	sonId = sons[t2];
	while (sonId != -1)
	{
		if (sonId > 0)
		{
			t2 = sonId + e2;
			e2 = 0;
		}
		else {
			int3 t = tris[t2];
			int3 tson1 = tris[-sonId];
			int tag = 2;
			if (t.x == tson1.x) tag = 0;
			if (t.y == tson1.x) tag = 1;

			if (e2 == tag)
			{
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (realStart == p2)
				{
					if (inner(p2, s2, p1) || inner(p2, p1, s2))
					{
						t2 = -sonId;
					}
					else {
						t2 = -sonId + 1;
					}
				}
				else {
					if (inner(p1, p2, realStart))
					{
						t2 = -sonId;
					}
					else {
						t2 = -sonId + 1;
					}
				}
				e2 = 0;
			}
			else if (e2 == (tag + 1) % 3)
			{
				t2 = -sonId + 1;
				e2 = 1;
			}
			else {
				t2 = -sonId;
				e2 = 2;
			}
		}
		sonId = sons[t2];
	}
}

__device__ int locateSegmentOnEdge(int3* tris, Point2d* points, const Point2d& realStart, const Point2d& s2, int* sons, int& t1, int& e1, int& t2, int& e2)
{
	int sonId = sons[t1];
	while (sonId != -1)
	{
		if (sonId > 0)
		{
			t1 = sonId + e1;
			e1 = 0;
		}
		else {
			int3 t = tris[t1];
			int3 tson1 = tris[-sonId];
			int tag = 2;
			if (t.x == tson1.x) tag = 0;
			if (t.y == tson1.x) tag = 1;

			if (e1 == tag)
			{
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (realStart == p2)
				{
					if (inner(p2, s2, p1) || inner(p2, p1, s2))
					{
						t1 = -sonId;
					}
					else {
						t1 = -sonId + 1;
						//printf("t1 %d\n", t1);
					}
				}
				else {
					if (inner(p1, p2, realStart))
					{
						t1 = -sonId;
					}
					else {
						t1 = -sonId + 1;
					}
				}
				e1 = 0;
			}
			else if (e1 == (tag + 1) % 3)
			{
				t1 = -sonId + 1;
				e1 = 1;
			}
			else {
				t1 = -sonId;
				e1 = 2;
			}
		}
		sonId = sons[t1];
	}

	sonId = sons[t2];
	while (sonId != -1)
	{
		{
			int3 t = tris[t2];
			Point2d p1, p2, p3;
			p1 = points[t.x];
			p2 = points[t.y];
			p3 = points[t.z];
			//printf("%d %d   %d %d  %d %d\n", p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
		}
		if (sonId > 0)
		{
			t2 = sonId + e2;
			e2 = 0;
		}
		else {
			int3 t = tris[t2];
			int3 tson1 = tris[-sonId];
			int tag = 2;
			if (t.x == tson1.x) tag = 0;
			if (t.y == tson1.x) tag = 1;

			if (e2 == tag)
			{
				Point2d p1 = points[tson1.y], p2 = points[tson1.z];
				if (realStart == p2)
				{
					//printf("here!!!!!\n");
					//printf("%d %d  %d %d  %d %d\n", p2.x, p2.y, s2.x, s2.y, p1.x, p1.y);
					if (inner(p2, s2, p1) || inner(p2, p1, s2))
					{
						t2 = -sonId;
					}
					else {
						t2 = -sonId + 1;
					}
				}
				else {
					if (inner(p1, p2, realStart))
					{
						t2 = -sonId;
					}
					else {
						t2 = -sonId + 1;
					}
				}
				e2 = 0;
			}
			else if (e2 == (tag + 1) % 3)
			{
				t2 = -sonId + 1;
				e2 = 1;
			}
			else {
				t2 = -sonId;
				e2 = 2;
			}
		}
		sonId = sons[t2];
	}
	{
		//int3 t = tris[t2];
		//Point2d p1, p2, p3;
		//p1 = points[t.x];
		//p2 = points[t.y];
		//p3 = points[t.z];
		//printf("%d %d   %d %d  %d %d\n", p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
	}
}
//458257610 189340955   458257370 189340955   0

//return +x : located on the triangle X
//return -y : located on the edge [Y % 3] of triangle [Y / 3]
__device__ int locateSegmentFirst(const Point2d& s1, const Point2d& s2, const HomoPoint2d& realStart,
	Point2d* points, int3* tris, int* sons, PointInteger res)
{
	Point2d RD = make_Point2d(res, 0);
	Point2d LT = make_Point2d(0, res);
	int o1 = orient2d(RD, LT, realStart);
	int o2 = orient2d(RD, LT, s2);
	int t1 = -1, t2, e1, e2;
	int firstTri = 0;
	if (o1 == 0 && o2 == 0) {
		t1 = 0; t2 = 1;
		e1 = 0; e2 = 1;
	}
	else {
		if (o1 < 0 || (o1 == 0) && (o2 < 0))
		{
			firstTri = 1;
		}
		int sid = sons[firstTri];
		while (sid != -1)
		{
			int3 t = tris[firstTri];
			Point2d cp;
			int3 st;
			if (sid > 0) {
				st = tris[sid];
				cp = points[st.x];
			}
			else {
				st = tris[-sid];
				cp = points[st.z];
			}
			int r = intSegmentTriangleFirst(s1, s2, realStart, points[t.x], points[t.y], points[t.z], cp);
			int ar = 0;
			
			if (sid < 0)
			{
				int tag = 2;
				if (t.x == st.x) tag = 0;
				if (t.y == st.x) tag = 1;

				if (r == -(tag + 1))
				{
					t1 = -sid; t2 = -sid + 1;
					e1 = 1; e2 = 2;
					break;
				}
				else if (r % 3 == tag)
				{
					ar = 0;
				}
				else {
					ar = 1;
				}
				firstTri = -sid + ar;
			}
			else {
				if (r > 0)
				{
					ar = r - 1;
				}
				else {
					t1 = sid + ((-r + 1) % 3); t2 = sid + ((-r) % 3);
					e1 = 2; e2 = 1;
					break;
				}
				firstTri = sid + ar;
			}
			sid = sons[firstTri];
		}
	}
	if (t1 != -1) {
		locateSegmentOnEdge(tris, points, realStart, s2, sons, t1, e1, t2, e2);
		return t1 < t2 ? -(t1 * 3 + e1) : -(t2 * 3 + e2);
	}
	else {
		return firstTri;
	}
}

__device__ int locateSegmentFirst(const Point2d& s1, const Point2d& s2, const Point2d& realStart,
	Point2d* points, int3* tris, int* sons, PointInteger res)
{
	Point2d RD = make_Point2d(res, 0);
	Point2d LT = make_Point2d(0, res);
	int o1 = orient2d(RD, LT, realStart);
	int o2 = orient2d(RD, LT, s2);
	int t1 = -1, t2, e1, e2;
	int firstTri = 0;
	if (o1 == 0 && o2 == 0) {
		t1 = 0; t2 = 1;
		e1 = 0; e2 = 1;
	}
	else {
		if (o1 < 0 || (o1 == 0) && (o2 < 0))
		{
			firstTri = 1;
		}
		int sid = sons[firstTri];
		int loopCount = 0;
		//printf("first tri %d\n", firstTri);
		while (sid != -1)
		{
			int3 t = tris[firstTri];
			Point2d cp;
			int3 st;
			if (sid > 0) {
				st = tris[sid];
				cp = points[st.x];
			}
			else {
				st = tris[-sid];
				cp = points[st.z];
			}
			int r = intSegmentTriangleFirst(s1, s2, realStart, points[t.x], points[t.y], points[t.z], cp);
			int ar = 0;
			
			if (sid < 0)
			{
				int tag = 2;
				if (t.x == st.x) tag = 0;
				if (t.y == st.x) tag = 1;

				if (r == -(tag + 1))
				{
					t1 = -sid; t2 = -sid + 1;
					e1 = 1; e2 = 2;
					break;
				}
				else if (r % 3 == tag)
				{
					ar = 0;
				}
				else {
					ar = 1;
				}
				firstTri = -sid + ar;
			}
			else {
				if (r > 0)
				{
					ar = r - 1;
				}
				else {
					t1 = sid + ((-r + 1) % 3); t2 = sid + ((-r) % 3);
					e1 = 2; e2 = 1;
					break;
				}
				firstTri = sid + ar;
			}
			//printf("%d %d %d\n", sid, ar, firstTri);
			sid = sons[firstTri];
		}
	}
	if (t1 != -1) {
		locateSegmentOnEdge(tris, points, realStart, s2, sons, t1, e1, t2, e2);
		return t1 < t2 ? -(t1 * 3 + e1) : -(t2 * 3 + e2);
	}
	else {
		return firstTri;
	}
}

__device__ int findInGrid(const HomoPoint2d& p, Point2d q, Point2d* points, int3* tris, int* bitmapOffset, int* bitmapTris, int ad)
{
	int L = bitmapOffset[ad];
	int R = bitmapOffset[ad + 1];
	for (int i = L; i < R; i++)
	{
		int ttId = bitmapTris[i];
		int3 t = tris[ttId];
		Point2d x = points[t.x], y = points[t.y], z = points[t.z];
		//printf("%d %d  %d %d  %d %d\n", x.x, x.y, y.x, y.y, z.x, z.y);
		int o1 = orient2d(x, y, p);
		int o2 = orient2d(y, z, p);
		int o3 = orient2d(z, x, p);
		if (o1 >= 0 && o2 >= 0 && o3 >= 0)
		{
			bool flag = true;
			if (o1 == 0)
			{
				if (orient2d(x, y, q) < 0) flag = false;
			}
			if (o2 == 0)
			{
				if (orient2d(y, z, q) < 0) flag = false;
			}
			if (o3 == 0)
			{
				if (orient2d(z, x, q) < 0) flag = false;
			}
			if (flag) {
				return ttId;
			}
		}
	}
	printf("not find\n");
	return -1;
}
__device__ int findInBitmap(const HomoPoint2d& p, Point2d q, Point2d* points, int3* tris, int* bitmapOffset, int* bitmapTris)
{
	int orderx, ordery;
	long long tx = p.x >> (logPointRes - logBitmap);
	long long ty = p.y >> (logPointRes - logBitmap);
	orderx = tx / p.z;
	ordery = ty / p.z;
	//orderx = tx >> (logPointRes - logBitmap);
	//ordery = ty >> (logPointRes - logBitmap);
	int ad = orderx * (1 << logBitmap) + ordery;
	//if (ad == 0)
	//{
	//	printf("%d %d %lld %lld\n", orderx, ordery, tx, ty);
	//}
	return findInGrid(p, q, points, tris, bitmapOffset, bitmapTris, ad);
}
__device__ bool inside(const HomoPoint2d& p, Point2d* points, int3* tris, int i)
{
	int3 t = tris[i];
	Point2d p1 = points[t.x];
	Point2d p2 = points[t.y];
	Point2d p3 = points[t.z];
	return orient2d(p1, p2, p) >= 0 && orient2d(p2, p3, p) >= 0 && orient2d(p3, p1, p) >= 0;
}
__device__ int locatePointSimple(const HomoPoint2d& startP, Point2d s2, Point2d* points, int3* tris, int* sons, int* bitmapOffset, int* bitmapTris)
{
	int i = findInBitmap(startP, s2, points, tris, bitmapOffset, bitmapTris);

	//if (!inside(startP, points, tris, i))
	//{
	//	printf("not inside\n");
	//}
	int sid = sons[i];
	while (sid != -1)
	{
		if (sid < 0)
		{
			if (inside(startP, points, tris, -sid))
			{
				i = -sid;
			}
			else {
				i = -sid + 1;
			}
		}
		else {
			if (inside(startP, points, tris, sid))
			{
				i = sid;
			}
			else if (inside(startP, points, tris, sid + 1)) {
				i = sid + 1;
			}
			else {
				i = sid + 2;
			}
		}
		sid = sons[i];
	}

	
	return i;
}
__global__ void locatePointsLB_s3_boost(Point2d* points, int3* tris, int* sons, int2* cons, int* bitmapOffset, int* bitmapTris, int* numSegs, int* sumSegs, int2* lbsAnswer, int* firstTriIds, int numCons, int numTh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTh) return;

	int conId = LBS(sumSegs, 0, numCons, tid);
	int order = tid - sumSegs[conId];
	lbsAnswer[tid] = make_int2(conId, order);
	int2 con = cons[conId];

	HomoPoint2d startP;
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];
	
	if (order != 0) {
		
		firstTriIds[tid] = locatePointSimple(make_homoPoint(s1, s2, order, numSegs[conId]),
			s2, points, tris, sons, bitmapOffset, bitmapTris);
		//firstTriIds[tid] = locatePointSimple(make_homoPoint(s1, s2, 0, 1),
		//	s2, points, tris, sons, bitmapOffset, bitmapTris);
	}
	else {
		if (numSegs[conId] == 1)
		{
			firstTriIds[tid] = locatePointSimple(make_homoPoint(s1, s2, 1, 2),
				s2, points, tris, sons, bitmapOffset, bitmapTris);
			//firstTriIds[tid] = -1;
		}
		else {
			firstTriIds[tid] = -1;
		}
	}
}

__global__ void locateStart(Point2d* points, int3* tris, int3* adjTris, int2* cons, int* sumSegs, int* firstTriIds, int* ignores, int numCons)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numCons) return;

	int id = sumSegs[tid];
	int triId = firstTriIds[id];
	if (triId == -1)
	{
		triId = firstTriIds[id + 1];
	}
	if (triId == -2) return;
	int2 con = cons[tid];
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];

	int3 t = tris[triId];
	int count = 0;
	while (!contains(t, con.x))
	{
		count++;
		if (count == 10000)
		{
			//printf("igrnore a\n");
			ignores[tid] = 1;
			break;
		}
		int3 at = adjTris[triId];

		if (con.y == t.x) triId = at.x;
		else if (con.y == t.y) triId = at.y;
		else if (con.y == t.z) triId = at.z;
		else {
			Point2d p1 = points[t.x];
			Point2d p2 = points[t.y];
			Point2d p3 = points[t.z];
			int o1 = orient2d(s2, s1, p1);
			int o2 = orient2d(s2, s1, p2);
			int o3 = orient2d(s2, s1, p3);
			int o12 = orient2d(p1, p2, s1);
			int o23 = orient2d(p2, p3, s1);
			//if (o1 == 0 || o2 == 0 || o3 == 0)
			//{
			//	ignores[tid] = 1;
			//	break;
			//}
			if (o12 < 0 && o1 < 0 && o2 > 0)
			{
				triId = at.z;
			}
			else if (o23 < 0 && o2 < 0 && o3 > 0)
			{
				triId = at.x;
			}
			else {
				triId = at.y;
			}
		}
		t = tris[triId];
	}
	firstTriIds[id] = triId;
	if (contains(t, con.y)) {
		ignores[tid] = 2;
 	}
}

__global__ void locatePointsLB_s3(Point2d* points, int3* tris, int* sons, int2* cons, int* numSegs, int* sumSegs, int2* lbsAnswer, int* firstTriIds, PointInteger res, int numCons, int numTh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTh) return;

	int conId = LBS(sumSegs, 0, numCons, tid);
	int order = tid - sumSegs[conId];
	lbsAnswer[tid] = make_int2(conId, order);
	int2 con = cons[conId];

	HomoPoint2d startP;
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];
	if (order != 0) {
		startP = make_homoPoint(s1, s2, order, numSegs[conId]);
	}
	else {
		startP = make_homoPoint(s1);
	}

	int first = locateSegmentFirst(s1, s2, startP, points, tris, sons, res);
	firstTriIds[tid] = first;
}

__device__ int getNextTriangle(Point2d* points, int3* tris, int3* adjTris, int startTri, const Point2d& s1, const Point2d& s2)
{
	int nextTri;

	int3 t = tris[startTri];
	int3 at = adjTris[startTri];
	Point2d p1 = points[t.x];
	Point2d p2 = points[t.y];
	Point2d p3 = points[t.z];

	if (s2 == p1 || s2 == p2 || s2 == p3)
	{
		return -1;
	}

	int o1 = orient2d(s1, s2, p1);
	int o2 = orient2d(s1, s2, p2);
	int o3 = orient2d(s1, s2, p3);
	int o12 = orient2d(p1, p2, s2);
	int o23 = orient2d(p2, p3, s2);
	int o31 = orient2d(p3, p1, s2);

	if (s1 == p1) return at.x;
	if (s1 == p2) return at.y;
	if (s1 == p3) return at.z;

	if (o12 < 0 && o1 < 0 && o2 > 0)
	{
		return at.z;
	}
	if (o23 < 0 && o2 < 0 && o3 > 0)
	{
		return at.x;
	}
	if (o31 < 0 && o3 < 0 && o1 > 0)
	{
		return at.y;
	}
	
	if (o1 == 0) {
		return -t.x;
	}
	if (o2 == 0) {
		return -t.y;
	}
	return -t.z;
}

__device__ int getNextTriangle(Point2d* points, int3* tris, int* sons, int startTri, const Point2d& s1, const Point2d& s2, const PointInteger res)
{
	int tid = (-startTri) / 3;
	int eid = (-startTri) % 3;
	int pid1, pid2;
	int3 t = tris[tid];

	pid1 = getInt3(&t, (eid + 1) % 3);
	pid2 = getInt3(&t, (eid + 2) % 3);

	Point2d p1 = points[pid1];
	Point2d p2 = points[pid2];
	//printf("%d %d\n", pid1, pid2);
	//printf("%d %d  %d %d  %d %d\n", p1.x, p1.y, p2.x, p2.y, s2.x, s2.y);
	if (inner(p1, s2, p2))
	{
		pid1 = pid2;
		p1 = p2;
	}
	if (p1 == s2)
	{
		return -1;
	}
	//printf("%d\n", -pid1);
	return -pid1;
}

__global__ void overlapTriangleDetect(Point2d* points, int3* tris, int3* adjTris, int2* cons, int2* lbsAnswer, int* startTriIds, int* ignores, int* counts, int* sons, int numTh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTh) return;
	//if (tid != 58277) return;

	int2 answer = lbsAnswer[tid];
	int conId = answer.x; int order = answer.y;
	if (ignores[conId] == 1)
	{
		counts[tid] = 0;
		return;
	}
	int2 con = cons[conId];
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];
	int3 startP;

	int triId = startTriIds[tid];
	int endTri = -1;
	if (tid + 1 < numTh && lbsAnswer[tid + 1].x == conId) {
		endTri = startTriIds[tid + 1];
	}
	
	int3 t = tris[triId];
	int count = 0;
	while (triId != endTri)
	{
		count++;
		if (count > 100000) {
			ignores[conId] = 1;
			printf("igrnore\n");
			break;
		}
		if (contains(t, con.y)) break;
		int3 at = adjTris[triId];

		if (con.x == t.x) triId = at.x;
		else if (con.x == t.y) triId = at.y;
		else if (con.x == t.z) triId = at.z;
		else {
			Point2d p1 = points[t.x];
			Point2d p2 = points[t.y];
			Point2d p3 = points[t.z];
			int o1 = orient2d(s1, s2, p1);
			int o2 = orient2d(s1, s2, p2);
			int o3 = orient2d(s1, s2, p3);
			int o12 = orient2d(p1, p2, s2);
			int o23 = orient2d(p2, p3, s2);
			if (o1 == 0 || o2 == 0 || o3 == 0) {
				ignores[conId] = 1;
				break;
			}
			if (o12 < 0 && o1 < 0 && o2 > 0)
			{
				triId = at.z;
			}
			else if (o23 < 0 && o2 < 0 && o3 > 0)
			{
				triId = at.x;
			}
			else {
				triId = at.y;
			}
		}
		t = tris[triId];
	}
	counts[tid] = count;
}
__global__ void overlapTriangleCount(Point2d* points, int3* tris, int3* adjTris, int2* cons, int2* lbsAnswer, int* startTriIds, int* ignores, int* counts, int* sons, int numTh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTh) return;
	//if (tid != 58277) return;

	int2 answer = lbsAnswer[tid];
	int conId = answer.x; int order = answer.y;
	if (ignores[conId] != 0)
	{
		counts[tid] = 0;
		return;
	}
	int2 con = cons[conId];
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];
	int3 startP;

	int triId = startTriIds[tid];
	int endTri = -1;
	if (tid + 1 < numTh && lbsAnswer[tid + 1].x == conId) {
		endTri = startTriIds[tid + 1];
	}
	/*
	//printf("%d %d   %d %d   %d\n", s1.x, s1.y, s2.x, s2.y, order);
	int count = 0;
	while (startTri != endTri)
	{
		//printf("tri id %d\n", startTri);
		if (startTri > 0) {
			count++;
			startTri = getNextTriangle(points, tris, adjTris, startTri, s1, s2);
		}
		else {
			startTri = getNextTriangle(points, tris, sons, startTri, s1, s2, res);
		}
		if (startTri == END_STATE)
		{
			break;
		}
		if (startTri < 0) {
			//printf("locate %d %d\n", points[-startTri].x, points[-startTri].y);
			startTri = locateSegmentFirst(s1, s2, points[-startTri], points, tris, sons, res);
		}
	}
	*/

	int3 t = tris[triId];
	int count = 0;
	while (triId != endTri)
	{
		count++;
		if (contains(t, con.y)) break;
		int3 at = adjTris[triId];

		if (con.x == t.x) triId = at.x;
		else if (con.x == t.y) triId = at.y;
		else if (con.x == t.z) triId = at.z;
		else {
			Point2d p1 = points[t.x];
			Point2d p2 = points[t.y];
			Point2d p3 = points[t.z];
			int o1 = orient2d(s1, s2, p1);
			int o2 = orient2d(s1, s2, p2);
			int o3 = orient2d(s1, s2, p3);
			int o12 = orient2d(p1, p2, s2);
			int o23 = orient2d(p2, p3, s2);
			//if (o1 == 0 || o2 == 0 || o3 == 0) printf("errror\n");
			if (o12 < 0 && o1 < 0 && o2 > 0)
			{
				triId = at.z;
			}
			else if (o23 < 0 && o2 < 0 && o3 > 0)
			{
				triId = at.x;
			}
			else {
				triId = at.y;
			}
		}
		t = tris[triId];
	}
	counts[tid] = count;
}

__global__ void overlapTriangle(Point2d* points, int3* tris, int3* adjTris, int2* cons, int2* lbsAnswer, int* sons, int* startTriIds, int* ignores, int* offsets, int* triIds, int* conIds, int* orders, int* counts, int numTh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTh) return;
	//if (tid != 58277) return;

	int2 answer = lbsAnswer[tid];
	int conId = answer.x; int order = answer.y;
	if (ignores[conId] != 0)
	{
		return;
	}
	int2 con = cons[conId];
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];
	int3 startP;

	int triId = startTriIds[tid];
	int endTri = -1;
	if (tid + 1 < numTh && lbsAnswer[tid + 1].x == conId) {
		endTri = startTriIds[tid + 1];
	}

	int3 t = tris[triId];
	int count = 0;
	int offset = offsets[tid];
	while (triId != endTri)
	{
		int ad = offset + count;
		count++;
		triIds[ad] = triId;
		conIds[ad] = conId;
		orders[ad] = atomicAdd(&counts[triId], 1);

		if (contains(t, con.y)) break;
		int3 at = adjTris[triId];

		if (con.x == t.x) triId = at.x;
		else if (con.x == t.y) triId = at.y;
		else if (con.x == t.z) triId = at.z;
		else {
			Point2d p1 = points[t.x];
			Point2d p2 = points[t.y];
			Point2d p3 = points[t.z];
			int o1 = orient2d(s1, s2, p1);
			int o2 = orient2d(s1, s2, p2);
			int o3 = orient2d(s1, s2, p3);
			int o12 = orient2d(p1, p2, s2);
			int o23 = orient2d(p2, p3, s2);
			if (o12 < 0 && o1 < 0 && o2 > 0)
			{
				triId = at.z;
			}
			else if (o23 < 0 && o2 < 0 && o3 > 0)
			{
				triId = at.x;
			}
			else {
				triId = at.y;
			}
		}
		t = tris[triId];
	}
}

__global__ void findIntersectedCons(int* offsets, int* triIds, int* conIds, int* orders, int* intConsList, int numOT)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numOT) return;
	int triId = triIds[tid];
	int conId = conIds[tid];
	int order = orders[tid];

	intConsList[offsets[triId] + order] = conId;
}

__global__ void reduceCons(Point2d* points, int3* tris, int2* cons, int* sons, int* offsets, int* conList, int3* nearCons, int3* farCons, int numTriangles)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numTriangles) return;

	int3 nearCon = make_int3(-1, -1, -1);
	int3 farCon = make_int3(-1, -1, -1);
	int L = offsets[tid];
	int R = offsets[tid + 1];
	if (L == R) {
		return;
	}
	int3 t = tris[tid];
	Point2d px, py, pz;
	px = points[t.x];
	py = points[t.y];
	pz = points[t.z];

	for (int i = L; i < R; i++)
	{
		int ocid = conList[i];
		int2 oc = cons[ocid];

		Point2d t1 = points[oc.x];
		Point2d t2 = points[oc.y];
		int ox, oy, oz;
		ox = orient2d(t1, t2, px);
		oy = orient2d(t1, t2, py);
		oz = orient2d(t1, t2, pz);

		if (ox != 0 && ox != oy && ox != oz)
		{
			if (nearCon.x == -1)
			{
				nearCon.x = ocid;
			}
			else {
				Point2d s1, s2;
				{
					int2 tempCon = cons[nearCon.x];
					s1 = points[tempCon.x];
					s2 = points[tempCon.y];
				}
				int os = orient2d(s1, s2, px);
				if ((orient2d(s1, s2, t1) * os >= 0 && orient2d(s1, s2, t2) * os >= 0) ||
					(orient2d(t1, t2, s1) * ox <= 0 && orient2d(t1, t2, s2) * ox <= 0))
				{
					nearCon.x = ocid;
				}
			}

			if (oy != 0 && oz != 0) {
				if (farCon.x == -1)
				{
					farCon.x = ocid;
				}
				else {
					Point2d s1, s2;
					{
						int2 tempCon = cons[farCon.x];
						s1 = points[tempCon.x];
						s2 = points[tempCon.y];
					}
					int os = orient2d(s1, s2, px);
					if ((orient2d(t1, t2, s1) * ox >= 0 && orient2d(t1, t2, s2) * ox >= 0) ||
						(orient2d(s1, s2, t1) * os <= 0 && orient2d(s1, s2, t2) * os <= 0))
					{
						farCon.x = ocid;
					}
				}
			}
		}

		if (oy != 0 && oy != ox && oy != oz)
		{
			if (nearCon.y == -1)
			{
				nearCon.y = ocid;
			}
			else {
				Point2d s1, s2;
				{
					int2 tempCon = cons[nearCon.y];
					s1 = points[tempCon.x];
					s2 = points[tempCon.y];
				}
				int os = orient2d(s1, s2, py);
				if ((orient2d(s1, s2, t1) * os >= 0 && orient2d(s1, s2, t2) * os >= 0) ||
					(orient2d(t1, t2, s1) * oy <= 0 && orient2d(t1, t2, s2) * oy <= 0))
				{
					nearCon.y = ocid;
				}
			}
			if (ox != 0 && oz != 0) {
				if (farCon.y == -1)
				{
					farCon.y = ocid;
				}
				else {
					Point2d s1, s2;
					{
						int2 tempCon = cons[farCon.y];
						s1 = points[tempCon.x];
						s2 = points[tempCon.y];
					}
					int os = orient2d(s1, s2, py);
					if ((orient2d(t1, t2, s1) * oy >= 0 && orient2d(t1, t2, s2) * oy >= 0) ||
						(orient2d(s1, s2, t1) * os <= 0 && orient2d(s1, s2, t2) * os <= 0))
					{
						farCon.y = ocid;
					}
				}
			}
		}

		if (oz != 0 && oz != ox && oz != oy)
		{
			if (nearCon.z == -1)
			{
				nearCon.z = ocid;
			}
			else {
				Point2d s1, s2;
				{
					int2 tempCon = cons[nearCon.z];
					s1 = points[tempCon.x];
					s2 = points[tempCon.y];
				}
				int os = orient2d(s1, s2, pz);
				if ((orient2d(s1, s2, t1) * os >= 0 && orient2d(s1, s2, t2) * os >= 0) ||
					(orient2d(t1, t2, s1) * oz <= 0 && orient2d(t1, t2, s2) * oz <= 0))
				{
					nearCon.z = ocid;
				}
			}

			if (ox != 0 && oy != 0) {
				if (farCon.z == -1)
				{
					farCon.z = ocid;
				}
				else {
					Point2d s1, s2;
					{
						int2 tempCon = cons[farCon.z];
						s1 = points[tempCon.x];
						s2 = points[tempCon.y];
					}
					int os = orient2d(s1, s2, pz);
					if ((orient2d(t1, t2, s1) * oz >= 0 && orient2d(t1, t2, s2) * oz >= 0) ||
						(orient2d(s1, s2, t1) * os <= 0 && orient2d(s1, s2, t2) * os <= 0))
					{
						farCon.z = ocid;
					}
				}
			}
		}
	}
	nearCons[tid] = nearCon;
	farCons[tid] = farCon;
	sons[tid] = 0;
}
/*
__global__ void formHalfStrip_s1(Point2d* points, int3* tris, int2* cons, int* triIds, int* conIds, int2* answers, int* counts, int numOT)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numOT) return;

	int triId = triIds[tid];
	int conId = conIds[tid];

	bool isFirst = ((tid == 0) || (conId != conIds[tid - 1]));
	bool isLast = ((tid == numOT - 1) || (conId != conIds[tid + 1]));

	int3 t = tris[triId];
	int2 answer{ -1, -1 };
	if (isFirst && isLast) {
		int3 tri = tris[triId];
		int2 con = cons[conId];
		if (tri.x != con.x && tri.x != con.y)
		{
			if (orient2d(points[con.x], points[con.y], points[tri.x]) > 0)
			{
				answer.x = tri.x;
			}
			else {
				answer.y = tri.x;
			}
		}
		if (tri.y != con.x && tri.y != con.y)
		{
			if (orient2d(points[con.x], points[con.y], points[tri.y]) > 0)
			{
				answer.x = tri.y;
			}
			else {
				answer.y = tri.y;
			}
		}
		if (tri.z != con.x && tri.z != con.y)
		{
			if (orient2d(points[con.x], points[con.y], points[tri.z]) > 0)
			{
				answer.x = tri.z;
			}
			else {
				answer.y = tri.z;
			}
		}
	}
	else if (isFirst)
	{
		int3 tri = tris[triId];
		int2 con = cons[conId];
		if (con.x != tri.x) {
			int j = judgePoint(points, tris, cons, offsets, conList, tri.x, triId, conId);
			if (j == 1)
			{
				answer.x = tri.x;
			}
			else if (j == -1)
			{
				answer.y = tri.x;
			}
		}
		if (con.x != tri.y) {
			int j = judgePoint(points, tris, cons, offsets, conList, tri.y, triId, conId);
			if (j == 1)
			{
				answer.x = tri.y;
			}
			else if (j == -1)
			{
				answer.y = tri.y;
			}
		}
		if (con.x != tri.z) {
			int j = judgePoint(points, tris, cons, offsets, conList, tri.z, triId, conId);
			if (j == 1)
			{
				answer.x = tri.z;
			}
			else if (j == -1)
			{
				answer.y = tri.z;
			}
		}
	}
	else if (isLast)
	{
		//do nothing?
	}
	else {
		int3 prevTri = tris[triIds[tid - 1]];
		int pid;
		if (t.x != prevTri.x && t.x != prevTri.y && t.x != prevTri.z) pid = t.x;
		if (t.y != prevTri.x && t.y != prevTri.y && t.y != prevTri.z) pid = t.y;
		if (t.z != prevTri.x && t.z != prevTri.y && t.z != prevTri.z) pid = t.z;
		int j = judgePoint(points, tris, cons, offsets, conList, pid, triId, conId);
		if (j == 1)
		{
			answer.x = pid;
		}
		else if (j == -1)
		{
			answer.y = pid;
		}
	}
	if (answer.x != -1) counts[tid] = 1; else counts[tid] = 0;
	if (answer.y != -1) counts[tid + numOT] = 1; else counts[tid + numOT] = 0;
	if (isFirst || isLast) {
		counts[tid]++;
		counts[tid + numOT]++;
	}
	answers[tid] = answer;
}
*/

double locatePointsBoost(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int2* cons,
	int* bitmapOffsets, int* bitmapTris,
	int2*& lbsAnswer, int*& firstTriIds, int*& ignoreCons, int numCons, int& numTh)
{
	double elapsed = 0.0;
	unsigned long long* LBCount;
	utils::malloc(LBCount, numCons);
	unsigned long long* LBWsum;
	utils::malloc(LBWsum, 1);
	utils::malloc(ignoreCons, numCons);
	utils::memset(ignoreCons, numCons);
	utils::memset(LBWsum, 1);
	elapsed += utils::kernel("s1", locatePointsLB_s1, numCons, 128, points, cons, LBCount, LBWsum, (int)logPointRes, numCons);
	
	int* LBSegs;
	utils::malloc(LBSegs, numCons + 1);
	elapsed += utils::kernel("s2", locatePointsLB_s2, numCons, 128, LBCount, LBSegs, LBWsum, (1 << logLBNumThreads), numCons);
	
	int* LBSegsSum;
	utils::malloc(LBSegsSum, numCons + 1);
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, LBSegs, LBSegsSum, numCons + 1);
	numTh = utils::getValue(LBSegsSum, numCons);
	printf("actual threads %d\n", numTh);

	utils::malloc(lbsAnswer, numTh);
	utils::malloc(firstTriIds, numTh);
	elapsed += utils::kernel("locatePointsLB_s3_boost", locatePointsLB_s3_boost, numTh, 128,
		points, tris, sons, cons, bitmapOffsets, bitmapTris, LBSegs, LBSegsSum, lbsAnswer, firstTriIds, numCons, numTh);
	
	elapsed += utils::kernel("find x", locateStart, numCons, 128, points, tris, adjTris, cons, LBSegsSum, firstTriIds, ignoreCons, numCons);
	utils::release(LBSegsSum, numCons + 1);
	utils::release(LBCount, numCons);
	utils::release(LBWsum, 1);
	utils::release(LBSegs, numCons + 1);
	return elapsed;
}

double locatePoints(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int* sons, int2* cons,
	int2*& lbsAnswer, int*& firstTriIds, int numCons, int& numTh)
{
	double elapsed = 0.0;
	unsigned long long* LBCount;
	utils::malloc(LBCount, numCons);
	unsigned long long* LBWsum;
	utils::malloc(LBWsum, 1);
	utils::memset(LBWsum, 1);
	elapsed += utils::kernel("s1", locatePointsLB_s1, numCons, 128, points, cons, LBCount, LBWsum, (int)logPointRes, numCons);

	int* LBSegs;
	utils::malloc(LBSegs, numCons + 1);
	utils::kernel("s2", locatePointsLB_s2, numCons, 128, LBCount, LBSegs, LBWsum, (1 << logLBNumThreads), numCons);

	int* LBSegsSum;
	utils::malloc(LBSegsSum, numCons + 1);
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, LBSegs, LBSegsSum, numCons + 1);
	numTh = utils::getValue(LBSegsSum, numCons);
	printf("actual threads %d\n", numTh);

	utils::malloc(lbsAnswer, numTh);
	utils::malloc(firstTriIds, numTh);
	//utils::kernel("locatePointsLB_s3_boost", locatePointsLB_s3_boost, numTh, 128,
	//	points, tris, sons, cons, bitmapOffsets, bitmapTris, LBSegs, LBSegsSum, lbsAnswer, firstTriIds, numCons, numTh);

	utils::release(LBSegsSum, numCons + 1);
	utils::release(LBCount, numCons);
	utils::release(LBWsum, 1);
	utils::release(LBSegs, numCons + 1);
	return elapsed;
}

__global__ void formPolygon_s1(Point2d* points, int3* tris, int2* cons, int* triIds, int* conIds, int3* nearCons, int3* farCons, int* pnCounts, int numI)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numI) return;

	int conId = conIds[tid];
	int2 con = cons[conId];
	Point2d s1 = points[con.x];
	Point2d s2 = points[con.y];
	int triId = triIds[tid];
	int3 t = tris[triId];

	if (contains(t, con.x))
	{
		int id = getIdInt3(t, con.x);
		if (getInt3(nearCons + triId, (id + 2) % 3) == conId)
		{
			pnCounts[tid] = 1;
		}
		else {
			pnCounts[tid] = 0;
		}
		if (getInt3(nearCons + triId, (id + 1) % 3) == conId)
		{
			pnCounts[tid + numI] = 1;
		}
		else {
			pnCounts[tid + numI] = 0;
		}
	} else if (contains(t, con.y))
	{
		pnCounts[tid] = 0;
		pnCounts[tid + numI] = 0;
	}
	else {
		int pos = 0;
		int neg = 0;
		int3 ot = tris[triIds[tid - 1]];
		Point2d p;
		int id;
		if (!contains(ot, t.x)) id = 0;
		else if (!contains(ot, t.y)) id = 1;
		else id = 2;

		p = points[getInt3(&t, id)];
		int o = orient2d(s1, s2, p) > 0 ? 2 : 1;

		int tempf = getInt3(farCons + triId, (id + o) % 3);
		if ((tempf < 0 || tempf == conId) && getInt3(nearCons + triId, id) < 0)
		{
			if (o == 2) pos = 1; else neg = 1;
		}

		pnCounts[tid] = pos;

		pnCounts[tid + numI] = neg;
	}
}

__global__ void formPolygon_s2(int3* tris, int2* cons, int* triIds, int* conIds, int* pnOffsets, int* pIds, int* cIds, int numI)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numI) return;

	int conId = conIds[tid];
	int2 con = cons[conId];
	int3 t = tris[triIds[tid]];

	int posOf = pnOffsets[tid];
	int negOf = pnOffsets[tid + numI];
	if (contains(t, con.x))
	{
		int id = getIdInt3(t, con.x);
		//pIds[posOf] = con.x;
		//cIds[posOf] = conId;
		//pIds[negOf] = con.x;
		//cIds[negOf] = conId;

		if (posOf + 1 == pnOffsets[tid + 1])
		{
			pIds[posOf] = getInt3(&t, (id + 2) % 3);
			cIds[posOf] = conId + 1;
		}
		if (negOf + 1 == pnOffsets[tid + numI + 1])
		{
			pIds[negOf] = getInt3(&t, (id + 1) % 3);
			cIds[negOf] = -(conId + 1);
		}
	}
	else if (contains(t, con.y))
	{
		//pIds[posOf] = con.y;
		//cIds[posOf] = conId;
		//pIds[negOf] = con.y;
		//cIds[negOf] = conId;
	}
	else {
		int3 ot = tris[triIds[tid - 1]];
		int nid;
		if (!contains(ot, t.x)) nid = t.x;
		else if (!contains(ot, t.y)) nid = t.y;
		else nid = t.z;
		if (posOf != pnOffsets[tid + 1])
		{
			pIds[posOf] = nid;
			cIds[posOf] = conId+1;
		}
		else if (negOf != pnOffsets[tid + numI + 1])
		{
			pIds[negOf] = nid;
			cIds[negOf] = -(conId + 1);
		}
	}
}

__global__ void calDistance(Point2d* points, int2* cons, int* pIds, int* cIds, MyInteger<logPointRes * 2 + 1>* leafs, int numN)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numN) return;

	MyInteger<logPointRes * 2 + 1> ret;
	int cid = cIds[tid];
	cid = cid > 0 ? cid - 1 : -cid - 1;
	int2 seg = cons[cid];

	Point2d p0 = points[pIds[tid]];
	Point2d p1 = points[seg.x];
	Point2d p2 = points[seg.y];
	Point2d d1{ p2.x - p1.x, p2.y - p1.y };
	Point2d d2{ p0.x - p1.x, p0.y - p1.y };
	ret = d2.x;
	ret *= d1.y;
	MyInteger<logPointRes * 2 + 1> temp = d1.x;
	temp *= d2.y;
	ret -= temp;

	//if (seg.x == pIds[tid]) { printf("same point\n"); }
	//if (ret == 0) printf("polygon end\n");
 	if (ret.isNegative()) ret = -ret;
	leafs[tid + nextPowerOfTwo(numN)] = ret;
}

__global__ void makeTree(MyInteger<logPointRes * 2 + 1>* leafs, int* cIds, int* atomicLabel, int numN)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numN) return;

	int cid = cIds[tid];
	int i = tid + nextPowerOfTwo(numN);
	MyInteger<logPointRes * 2 + 1> a = leafs[i];
	
	while (!a.isZero())
	{
		__threadfence();
		int label = atomicExch(atomicLabel + (i >> 1), cid);
		if (label == 0)
		{
			break;
		}
		else {
			if (label == cid) {
				if (leafs[i ^ 1] < a) a = leafs[i ^ 1];
			}
			else {
				a = 0;
			}
		}
		i >>= 1;
		leafs[i] = a;
	}
}


__device__ int searchOnTreeLeft(MyInteger<logPointRes * 2 + 1>* leafs, MyInteger<logPointRes * 2 + 1> a, int x)
{
	int base = nextPowerOfTwo(x) / 2;
	while ((x & 1) == 0 || ((x & 1) && leafs[x ^ 1] >= a)) x >>= 1;
	x -= 1;

	while (x < base)
	{
		if (leafs[x * 2 + 1] < a) x = x * 2 + 1;
		else x = x * 2;
	}
	return x;
}
__device__ int searchOnTreeRight(MyInteger<logPointRes * 2 + 1>* leafs, MyInteger<logPointRes * 2 + 1> a, int x)
{
	int base = nextPowerOfTwo(x) / 2;
	while ((x & 1) == 1 || ((x & 1) == 0 && leafs[x ^ 1] > a)) x >>= 1;
	x += 1;

	while (x < base)
	{
		if (leafs[x * 2] <= a) x = x * 2;
		else x = x * 2 + 1;
	}
	return x;
}

__global__ void reTriangle(int2* cons, MyInteger<logPointRes * 2 + 1>* leafs, int* pIds, int* cIds, int3* newTris, int numP)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= numP) return;
	int base = nextPowerOfTwo(numP);
	MyInteger<logPointRes * 2 + 1> a = leafs[tid + base];
	int cid = cIds[tid];
	//int l = tid + base - 1
	//int l = tid + base - 1;
	//while (leafs[l] >= a) l--;
	//int r = tid + base + 1;
	//while (leafs[r] > a) r++;
 	int l = searchOnTreeLeft(leafs, a, tid + nextPowerOfTwo(numP));
	int r = searchOnTreeRight(leafs, a, tid + nextPowerOfTwo(numP));
	int3 ret = make_int3(pIds[tid], pIds[l - base], pIds[r - base]);
	if (cIds[l - base] != cid)
	{
		ret.y = cons[cid > 0 ? cid - 1 : -cid - 1].x;
	}
	if (cIds[r - base] != cid)
	{
		ret.z = cons[cid > 0 ? cid - 1 : -cid - 1].y;
	}
	if (cid < 0)
	{
		int t = ret.y;
		ret.y = ret.z;
		ret.z = t;
	}
	newTris[tid] = ret;
}

double addConstraints(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int* bitmapOffsets, int* bitmapTris, int2* cons, int3*& newTris, int numCons, int numPoints, int numTriangles, int& numNewTris)
{
	double timecount = 0;
	int maxNumTh = (1 << logLBNumThreads) + numCons + 1;
	printf("boost max threads %d\n", maxNumTh);
	int* overlapCounts;
	utils::malloc(overlapCounts, maxNumTh);

	int2* lbsAnswer;
	int* firstTriIds;
	int numTh;
	int* noFlips;
	int* ignoreCons;
	utils::malloc(noFlips, numTriangles * 3);
	//locatePoints(cubTempStorage, cubTempStorageBytes, points, tris, sons, cons, lbsAnswer, firstTriIds, numCons, numTh);
	timecount += locatePointsBoost(cubTempStorage, cubTempStorageBytes, points, tris, adjTris, sons, cons,
		bitmapOffsets, bitmapTris, lbsAnswer, firstTriIds, ignoreCons, numCons, numTh);
	utils::kernel("overlap detect", overlapTriangleDetect, numTh, 128, points, tris, adjTris, cons, lbsAnswer, firstTriIds, ignoreCons, overlapCounts, sons, numTh);
	timecount += utils::kernel("overlap count", overlapTriangleCount, numTh, 128, points, tris, adjTris, cons, lbsAnswer, firstTriIds, ignoreCons, overlapCounts, sons, numTh);
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, overlapCounts, overlapCounts, numTh + 1);
	int overlapNum = utils::getValue(overlapCounts, numTh);
	{
		int* ignoreConsSum;
		utils::malloc(ignoreConsSum, numCons + 1);
		utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, ignoreCons, ignoreConsSum, numCons + 1);
		int sumig = utils::getValue(ignoreConsSum, numCons);
		printf("sum ig: %d ------------------------------------\n", sumig);
		utils::release(ignoreConsSum, numCons + 1);
	}
	//overlapTriangle();
	printf("overlap %d\n", overlapNum);

	int* overlapTriIds;
	utils::malloc(overlapTriIds, overlapNum);
	int* overlapOrders;
	int* overlapConIds;
	utils::malloc(overlapOrders, overlapNum);
	utils::malloc(overlapConIds, overlapNum);
	int* triangleIntCounts;
	utils::malloc(triangleIntCounts, numTriangles + 1);
	utils::memset(triangleIntCounts, numTriangles + 1);
	
	timecount += utils::kernel("overlap", overlapTriangle, numTh, 128, points, tris, adjTris, cons, lbsAnswer, sons, firstTriIds, ignoreCons, overlapCounts, overlapTriIds, overlapConIds, overlapOrders, triangleIntCounts, numTh);
	//overlapTriangle();

	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, triangleIntCounts, triangleIntCounts, numTriangles + 1);

	int consListLength = utils::getValue(triangleIntCounts, numTriangles);
	int* intConsList;
	printf("conList %d\n", consListLength);
	utils::malloc(intConsList, consListLength);
	timecount += utils::kernel("findIntersectedCons", findIntersectedCons, overlapNum, 128, triangleIntCounts, overlapTriIds, overlapConIds, overlapOrders, intConsList, overlapNum);

	int3* nearCons, * farCons;
	utils::malloc(nearCons, numTriangles);
	utils::malloc(farCons, numTriangles);
	timecount += utils::kernel("reduceCons", reduceCons, numTriangles, 128, points, tris, cons, sons, triangleIntCounts, intConsList, nearCons, farCons, numTriangles);
	
	int* pnCounts;
	utils::malloc(pnCounts, overlapNum * 2 + 1);
	timecount += utils::kernel("form1", formPolygon_s1, overlapNum, 128, points, tris, cons, overlapTriIds, overlapConIds,
		nearCons, farCons, pnCounts, overlapNum);
	//加入新三角形
	utils::exlusiveScan(cubTempStorage,  cubTempStorageBytes, pnCounts, pnCounts, overlapNum * 2 + 1);
	
	int numPolyP = utils::getValue(pnCounts, overlapNum * 2);
	printf("polygons %d\n", numPolyP);

	utils::release(nearCons, numTriangles);
	utils::release(farCons, numTriangles);

	int* polyP, * polyC;
	utils::malloc(polyP, numPolyP);
	utils::malloc(polyC, numPolyP);
	timecount += utils::kernel("form2", formPolygon_s2, overlapNum, 128, tris, cons, overlapTriIds, overlapConIds, pnCounts, polyP, polyC, overlapNum);
	
	utils::release(pnCounts, overlapNum * 2 + 1);
	MyInteger<logPointRes * 2 + 1>* leafs;
	utils::malloc(leafs, nextPowerOfTwo(numPolyP) * 2);
	utils::memset(leafs, nextPowerOfTwo(numPolyP) * 2);
	timecount += utils::kernel("cal", calDistance, numPolyP, 128, points, cons, polyP, polyC, leafs, numPolyP);

	int* atomicLabel;
	utils::malloc(atomicLabel, nextPowerOfTwo(numPolyP) * 2);
	utils::memset(atomicLabel, nextPowerOfTwo(numPolyP) * 2);
	timecount += utils::kernel("makeTree", makeTree, numPolyP, 128, leafs, polyC, atomicLabel, numPolyP);
	utils::release(atomicLabel, nextPowerOfTwo(numPolyP) * 2);

	numNewTris = numPolyP;
	utils::malloc(newTris, numPolyP);
	timecount += utils::kernel("reTriangle", reTriangle, numPolyP, 128, cons, leafs, polyP, polyC, newTris, numPolyP);
	//drawOverlaps(points, newTris, firstTriIds, numPolyP, numPoints, numPolyP);
	//drawPolygons(points, polyP, polyC, numPoints, numPolyP);

	utils::release(polyP, numPolyP);
	utils::release(polyC, numPolyP);
	utils::release(leafs, nextPowerOfTwo(numPolyP) * 2);
	//utils::release(nearCons, numTriangles);
	//utils::release(farCons, numTriangles);

	utils::release(intConsList, consListLength);
	utils::release(overlapCounts, maxNumTh);

	utils::release(lbsAnswer, numTh);
	utils::release(firstTriIds, numTh);

	utils::release(overlapTriIds, overlapNum);
	utils::release(overlapOrders, overlapNum);
	utils::release(overlapConIds, overlapNum);
	utils::release(triangleIntCounts, numTriangles + 1);
	printf("constraints %f\n", timecount);
	return timecount;
}

double addConstraints(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int2* cons, int numCons, int numPoints, int numTriangles)
{
	return 0;
	/*
	int maxNumTh = (1 << logLBNumThreads) + numCons + 1;
	printf("max threads %d\n", maxNumTh);
	int* overlapCounts;
	utils::malloc(overlapCounts, maxNumTh);

	int2* lbsAnswer;
	int* firstTriIds;
	int numTh;
	locatePoints(cubTempStorage, cubTempStorageBytes, points, tris, sons, cons, lbsAnswer, firstTriIds, numCons, numTh);

	utils::kernel("overlap count", overlapTriangleCount, numTh, 128, points, tris, adjTris, cons, lbsAnswer, firstTriIds, overlapCounts, sons, res, numTh);
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, overlapCounts, overlapCounts, numTh + 1);
	int overlapNum = utils::getValue(overlapCounts, numTh);

	//overlapTriangle();
	printf("overlap %d\n", overlapNum);

	int* overlapTriIds;
	utils::malloc(overlapTriIds, overlapNum);
	int* overlapOrders;
	int* overlapConIds;
	utils::malloc(overlapOrders, overlapNum);
	utils::malloc(overlapConIds, overlapNum);
	int* triangleIntCounts;
	utils::malloc(triangleIntCounts, numTriangles + 1);
	utils::memset(triangleIntCounts, numTriangles + 1);
	int* noFlips;
	utils::malloc(noFlips, numTriangles * 3);
	//utils::kernel("overlap", overlapTriangle, numTh, 128, points, tris, adjTris, cons, lbsAnswer, sons, firstTriIds, overlapCounts, overlapTriIds, overlapConIds, overlapOrders, triangleIntCounts, noFlips, res, numTh);
	//overlapTriangle();
	
	utils::exlusiveScan(cubTempStorage, cubTempStorageBytes, triangleIntCounts, triangleIntCounts, numTriangles + 1);

	int consListLength = utils::getValue(triangleIntCounts, numTriangles);
	int* intConsList;
	printf("conList %d\n", consListLength);
	utils::malloc(intConsList, consListLength);
	//utils::kernel("findIntersectedCons", findIntersectedCons, overlapNum, 128, triangleIntCounts, overlapTriIds, overlapConIds, overlapOrders, intConsList, overlapNum);

	int3* nearCons, *farCons;
	utils::malloc(nearCons, numTriangles);
	utils::malloc(farCons, numTriangles);
	//utils::kernel("reduceCons", reduceCons, numTriangles, 128, points, tris, cons, triangleIntCounts, intConsList, nearCons, farCons, numTriangles);
	
	
	drawOverlaps(points, tris, firstTriIds, numTh, numPoints, numTriangles);

	utils::release(nearCons, numTriangles);
	utils::release(farCons, numTriangles);
	utils::release(intConsList, consListLength);
	utils::release(overlapCounts, maxNumTh);

	utils::release(lbsAnswer, numTh);
	utils::release(firstTriIds, numTh);

	utils::release(overlapTriIds, overlapNum);
	utils::release(overlapOrders, overlapNum);
	utils::release(overlapConIds, overlapNum);
	utils::release(triangleIntCounts, numTriangles + 1);
	*/
}

//__device__ int intSegmentTriangleFirstDebug(const Point2d& s1, const Point2d& s2, const HomoPoint2d& h1,
//	const Point2d& t1, const Point2d& t2, const Point2d& t3, const Point2d& cp)
//{
//	int oc = orient2d(s1, s2, cp);
//	int o1c1 = orient2d(t1, cp, h1);
//	int o2c1 = orient2d(t2, cp, h1);
//	int o3c1 = orient2d(t3, cp, h1);
//
//	if ((o1c1 < 0 && o2c1 > 0) || (o1c1 == 0 && o2c1 > 0 && oc > 0) || (o1c1 < 0 && o2c1 == 0 && oc < 0)) {
//		printf("early 0\n");
//		return 0;
//	}
//
//	if ((o2c1 < 0 && o3c1 > 0) || (o2c1 == 0 && o3c1 > 0 && oc > 0) || (o2c1 < 0 && o3c1 == 0 && oc < 0)) {
//		return 1;
//	}
//
//	if ((o3c1 < 0 && o1c1 > 0) || (o3c1 == 0 && o1c1 > 0 && oc > 0) || (o3c1 < 0 && o1c1 == 0 && oc < 0))
//	{
//		return 2;
//	}
//
//	//oc == 0
//	int o1c2 = orient2d(t1, cp, s2);
//	int o2c2 = orient2d(t2, cp, s2);
//	int o3c2 = orient2d(t3, cp, s2);
//	printf("%d %d %d\n", o1c2, o2c2, o3c2);
//	if (h1 == cp) {
//		if (o1c2 < 0 && o2c2 > 0)
//		{
//			return 0;
//		}
//		if (o2c2 < 0 && o3c2 > 0)
//		{
//			return 1;
//		}
//		if (o3c2 < 0 && o1c2 > 0)
//		{
//			return 2;
//		}
//	}
//	if (o1c1 == 0 && o1c2 == 0) return 0;
//	if (o2c1 == 0 && o2c2 == 0) return 1;
//	if (o3c1 == 0 && o3c2 == 0) return 2;
//
//	return -1;
//}


//__device__ int locateSegmentFirstDebug(int2 s1, int2 s2, longlong3 realStart, int2* points, int3* tris, int* sons, int res)
//{
//	int2 RD = make_int2(res, 0);
//	int2 LT = make_int2(0, res);
//	int o1 = orient2d(RD, LT, realStart);
//	int o2 = orient2d(RD, LT, s2);
//	if (o1 == 0 && o2 == 0) return -1;
//	int firstTri = 0;
//	if (o1 < 0 || (o1 == 0) && (o2 < 0))
//	{
//		firstTri = 1;
//	}
//	int sid = sons[firstTri];
//	//printf("%d\n", sid);
//	while (sid != -1)
//	{
//		int3 t = tris[firstTri];
//		int2 t1 = points[t.x];
//		int2 t2 = points[t.y];
//		int2 t3 = points[t.z];
//		int2 cp = points[tris[sid].z];
//		printf("%d %d %d %d\n", t.x, t.y, t.z, tris[sid].z);
//		int r = intSegmentTriangleFirstDebug(s1, s2, realStart, t1, t2, t3, cp);
//		if (r == -1)
//		{
//			return -firstTri - 1;
//		}
//		//printf("%d %d\n", sid, r);
//		firstTri = sid + r;
//		sid = sons[firstTri];
//	}
//	printf("%d %d %d\n", tris[firstTri].x, tris[firstTri].y, tris[firstTri].z);
//	return firstTri;
//}