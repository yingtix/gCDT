#pragma once
#include "intX_t.cuh"
#include "math_base.h"

constexpr unsigned int nextPowerOfTwo(unsigned int x)
{
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

template<unsigned int N>
using MyInteger = JIO::Integer<nextPowerOfTwo(N) / 8, true>;

constexpr unsigned int logPointRes = 30;
//typedef int PointInteger;
using PointInteger = MyInteger<logPointRes>;
constexpr unsigned int logLBNumThreads = 20; //20
constexpr bool debug = false;
constexpr unsigned int logSample = 8;
constexpr unsigned int logBitmap = 10;
constexpr unsigned int rootBitmap = 1024;

//typedef longlong2 Point2d;
typedef int2 Point2d;

//struct Point2d
//{
//	  PointInteger x, y;
//};
struct Point2d2
{
	Point2d x, y;
};
__device__ inline Point2d make_Point2d(const PointInteger& _x, const PointInteger& _y)
{
	Point2d ret;
	ret.x = _x; ret.y = _y;
	return ret;
}

using HomoPointInteger = MyInteger<logPointRes + logLBNumThreads>;
struct HomoPoint2d
{
	HomoPointInteger x, y;
	int z;
};
/**
	1: a b c  counterclockwise
	0: a b c  line
   -1: a b c  clockwise
*/

//inline __device__ int orient2d(const int2& a, const int2& b, const int2& c)
//{
//	MyInteger<logPointRes * 2> o1 = (a.x - c.x);
//	o1 *= (b.y - c.y);
//	MyInteger<logPointRes * 2> o2 = (a.y - c.y);
//	o2 *= (b.x - c.x);
//	o1 -= o2;
//	if (o1.isZero()) return 0; else if (o1.isNegative()) return -1; else return 1;
//}

inline __device__ __host__ int orient2d(const Point2d& a, const Point2d& b, const Point2d& c)
{
	MyInteger<logPointRes * 2> o1 = (a.x - c.x);
	o1 *= (b.y - c.y);
	MyInteger<logPointRes * 2> o2 = (a.y - c.y);
	o2 *= (b.x - c.x);
	if (o1 > o2) return 1; else if (o1 < o2) return -1; else return 0;
	//if (o1.isZero()) return 0; else if (o1.isNegative()) return -1; else return 1;
}

inline __device__ int orient2d(int2 a, int2 b, longlong3 c)
{
	int64_t t = c.z;
	int64_t o1 = (t * a.x) * b.y;
	t = b.x; t *= c.y;
	o1 += t;
	t = a.y; t *= c.x;
	o1 += t;

	t = c.z;
	int64_t o2 = (t * a.y) * b.x;
	t = b.y; t *= c.x;
	o2 += t;
	t = a.x; t *= c.y;
	o2 += t;

	o1 -= o2;
	if (o1 == 0) return 0; else if (o1 > 0) return 1; else return -1;
}

inline __device__ int orient2d(const Point2d& a, const Point2d& b, const HomoPoint2d& c)
{
	MyInteger<logPointRes * 2 + logLBNumThreads + 2> t = c.z;
	MyInteger<logPointRes * 2 + logLBNumThreads + 2> o1 = (t * a.x) * b.y;

	t = b.x; t *= c.y;
	o1 += t;
	t = a.y; t *= c.x;
	o1 += t;
	
	t = c.z;
	MyInteger<logPointRes * 2 + logLBNumThreads + 2> o2 = (t * a.y) * b.x;
	t = b.y; t *= c.x;
	o2 += t;
	t = a.x; t *= c.y;
	o2 += t;

	if (o1 > o2) { return 1; }
	else if (o1 == o2) { return 0; }
	else { return -1; }
}

using int128_t = JIO::Integer<16, true>;

inline __host__ __device__ void multi3(const PointInteger& a, const PointInteger& b, const MyInteger<logPointRes * 2 + 1>& c, MyInteger<logPointRes * 4 + 1>& ret)
{
	ret = a;
	ret = ret * b;
	ret = ret * c;
}

inline __host__ __device__ uint2 multiply(unsigned int a, unsigned int b)
{
	unsigned long long c = a;
	c *= b;
	uint2 ret;
	ret.x = (c & 0xFFFFFFFF);
	ret.y = c >> 32;
	return ret;
}

inline __device__ __host__ uint4 multiply(uint2 a, uint2 b)
{
	uint4 c;
	uint2 temp = multiply(a.x, b.x);
	c.x = temp.x;
	c.y = temp.y;

	temp = multiply(a.y, b.x);
	c.y += temp.x;
	if (c.y < temp.x) c.z = 1; else c.z = 0;
	c.z += temp.y;

	temp = multiply(a.x, b.y);
	c.y += temp.x;
	if (c.y < temp.x) temp.y++;
	c.z += temp.y;
	if (c.z < temp.y) c.w = 1; else c.w = 0;

	temp = multiply(a.y, b.y);
	c.z += temp.x;
	if (c.z < temp.x) temp.y++;
	c.w += temp.y;
	return c;
}

inline __device__ __host__ void addIn(uint4& a, uint4 b)
{
	a.x += b.x;
	unsigned int carry = a.x < b.x;
	a.y += b.y + carry;
	carry = a.y < b.y || (a.y == b.y && carry != 0);
	a.z += b.z + carry;
	carry = a.z < b.z || (a.z == b.z && carry != 0);
	a.w += b.w + carry;
}

inline __device__ __host__ void multi33(int a, int b, long long c, int sgn, uint4* ret)
{
	if (a < 0) {
		sgn = 1 - sgn; a = -a;
	}
	if (b < 0)
	{
		sgn = 1 - sgn; b = -b;
	}
	if (c < 0) {
		sgn = 1 - sgn; c = -c;
	}
	addIn(ret[sgn], multiply(multiply(a, b), uint2{ unsigned int(c & 0xFFFFFFFF), unsigned int(c >> 32) }));
}


inline __device__ __host__ void multi33(int a, int b, long long c, int128_t& ret)
{
	long long temp = a;
	temp *= b;
	ret = c;
	ret *= temp;
}

inline __host__ __device__ int inCircle_(const Point2d& a, const Point2d& b, const Point2d& c, const Point2d& p)
{
	int2 ap, bp, cp;
	ap.x = a.x - p.x;
	ap.y = a.y - p.y;
	bp.x = b.x - p.x;
	bp.y = b.y - p.y;
	cp.x = c.x - p.x;
	cp.y = c.y - p.y;

	long long temp;
	temp = ap.x;
	temp = temp * ap.x;
	long long aa = temp;
	temp = ap.y;
	temp = temp * ap.y;
	aa = aa + temp;

	temp = bp.x;
	temp = temp * bp.x;
	long long bb = temp;
	temp = bp.y;
	temp = temp * bp.y;
	bb = bb + temp;

	temp = cp.x;
	temp = temp * cp.x;
	long long cc = temp;
	temp = cp.y;
	temp = temp * cp.y;
	cc = cc + temp;

	/* det
	   | ap.x  ap.y  aa |
	   | bp.x  bp.y  bb |
	   | cp.x  cp.y  cc |
	*/
	uint4 sum[2] = { {0, 0, 0, 0},{0, 0, 0, 0} };
	multi33(ap.x, bp.y, cc, 0, sum);

	multi33(ap.x, cp.y, bb, 1, sum);

	multi33(ap.y, bp.x, cc, 1, sum);

	multi33(ap.y, cp.x, bb, 0, sum);

	multi33(bp.x, cp.y, aa, 0, sum);

	multi33(cp.x, bp.y, aa, 1, sum);

	if (sum[0].w > sum[1].w) { return 1; }
	else if (sum[0].w < sum[1].w) { return -1; }
	else
	{
		if (sum[0].z > sum[1].z) { return 1; }
		else if (sum[0].z < sum[1].z) { return -1; }
		else
		{
			if (sum[0].y > sum[1].y) { return 1; }
			else if (sum[0].y < sum[1].y) { return -1; }
			else {
				if (sum[0].x > sum[1].x) { return 1; }
				else if (sum[0].x < sum[1].x) { return -1; }
				else return 0;
			}
		}
	}
}
inline __host__ __device__ int inCircle(const Point2d& a, const Point2d& b, const Point2d& c, const Point2d& p)
{
	Point2d ap, bp, cp;
	ap.x = a.x - p.x;
	ap.y = a.y - p.y;
	bp.x = b.x - p.x;
	bp.y = b.y - p.y;
	cp.x = c.x - p.x;
	cp.y = c.y - p.y;
	/* det
	   | ap.x  ap.y  ap.x*ap.x+ap.y*ap.y |
	   | bp.x  bp.y  bp.x*bp.x+bp.y*bp.y |
	   | cp.x  cp.y  cp.x*cp.x+cp.y*cp.y |
	*/
	MyInteger<logPointRes * 2> temp;
	temp = ap.x;
	temp = temp * ap.x;
	MyInteger<logPointRes * 2 + 1> aa = temp;
	temp = ap.y;
	temp = temp * ap.y;
	aa = aa + temp;

	temp = bp.x;
	temp = temp * bp.x;
	MyInteger<logPointRes * 2 + 1> bb = temp;
	temp = bp.y;
	temp = temp * bp.y;
	bb = bb + temp;

	temp = cp.x;
	temp = temp * cp.x;
	MyInteger<logPointRes * 2 + 1> cc = temp;
	temp = cp.y;
	temp = temp * cp.y;
	cc = cc + temp;

	/* det
	   | ap.x  ap.y  aa |
	   | bp.x  bp.y  bb |
	   | cp.x  cp.y  cc |
	*/
	MyInteger<logPointRes * 4 + 4> sum;
	MyInteger<logPointRes * 4 + 1> part;
	multi3(ap.x, bp.y, cc, part);
	sum = part;

	multi3(ap.x, cp.y, bb, part);
	sum = sum - part;

	multi3(ap.y, bp.x, cc, part);
	sum = sum - part;

	multi3(ap.y, cp.x, bb, part);
	sum = sum + part;

	multi3(bp.x, cp.y, aa, part);
	sum = sum + part;

	multi3(cp.x, bp.y, aa, part);
	sum = sum - part;
	if (sum.isZero()) return 0; else if (sum.isNegative()) return -1; else return 1;
}
inline __device__ int inCircle__(const int2& a, const int2& b, const int2& c, const int2& p)
{
	int2 ap, bp, cp;
	ap.x = a.x - p.x;
	ap.y = a.y - p.y;
	bp.x = b.x - p.x;
	bp.y = b.y - p.y;
	cp.x = c.x - p.x;
	cp.y = c.y - p.y;
	/* det
	   | ap.x  ap.y  ap.x*ap.x+ap.y*ap.y |
	   | bp.x  bp.y  bp.x*bp.x+bp.y*bp.y |
	   | cp.x  cp.y  cp.x*cp.x+cp.y*cp.y |
	*/
	long long temp;
	temp = ap.x;
	temp = temp * ap.x;
	long long aa = temp;
	temp = ap.y;
	temp = temp * ap.y;
	aa = aa + temp;

	temp = bp.x;
	temp = temp * bp.x;
	long long bb = temp;
	temp = bp.y;
	temp = temp * bp.y;
	bb = bb + temp;

	temp = cp.x;
	temp = temp * cp.x;
	long long cc = temp;
	temp = cp.y;
	temp = temp * cp.y;
	cc = cc + temp;
	/* det
	   | ap.x  ap.y  aa |
	   | bp.x  bp.y  bb |
	   | cp.x  cp.y  cc |
	*/
	int128_t sum;
	int128_t part;
	multi33(ap.x, bp.y, cc, part);
	sum = part;

	multi33(ap.x, cp.y, bb, part);
	sum = sum - part;

	multi33(ap.y, bp.x, cc, part);
	sum = sum - part;

	multi33(ap.y, cp.x, bb, part);
	sum = sum + part;

	multi33(bp.x, cp.y, aa, part);
	sum = sum + part;

	multi33(cp.x, bp.y, aa, part);
	sum = sum - part;
	if (sum.isZero()) return 0; else if (sum.isNegative()) return -1; else return 1;
}

inline __device__ bool inner(const PointInteger& a, const PointInteger& b, const PointInteger& c)
{
	return (a <= c && c <= b) || (a >= c && c >= b);
}
inline __device__ bool inner(const Point2d& a, const Point2d& b, const Point2d& c)
{
	return inner(a.x, b.x, c.x) && inner(a.y, b.y, c.y);
}
inline __device__ bool inner(const PointInteger& a, const PointInteger& b, const HomoPointInteger& c, const PointInteger& d)
{
	HomoPointInteger ad = a;
	ad *= d;
	HomoPointInteger bd = b;
	bd *= d;
	return (ad <= c && c <= bd) || (ad >= c && c >= bd);
}
inline __device__ bool inner(const Point2d& a, const Point2d& b, const HomoPoint2d& c)
{
	return inner(a.x, b.x, c.x, c.z) && inner(a.y, b.y, c.y, c.z);
}
inline __device__ bool contains(int3 a, int b)
{
	return a.x == b || a.y == b || a.z == b;
}

inline __device__ bool operator ==(const Point2d& a, const Point2d& b)
{
	return (a.x == b.x && a.y == b.y);
}

inline __device__ bool operator == (const HomoPoint2d& x, const Point2d& y)
{
	return (x.x == (HomoPointInteger)x.z * y.x && x.y == (HomoPointInteger)x.z * y.y);
}

inline __device__ int getIdInt3(const int3& t, int a)
{
	if (t.x == a) return 0;
	if (t.y == a) return 1;
	return 2;
}

inline __device__ __host__ const int getInt2(const int2* x, int t)
{
	return ((int*)x)[t];
}

inline __device__ __host__ int& getInt2(int2* x, int t)
{
	return ((int*)x)[t];
}

inline __device__ __host__ Point2d& getPoint2d2_(Point2d2* p, int t)
{
	return ((Point2d*)p)[t];
}

inline __device__ __host__ const Point2d getPoint2d2(const Point2d2* p, int t)
{
	return ((Point2d*)p)[t];
}

inline __device__ __host__ int getInt3(const int3* x, int t)
{
	return ((int*)x)[t];
}
inline __device__ __host__ int& getInt3(int3* x, int t)
{
	return ((int*)x)[t];
}

inline __device__ __host__ int getInt4(const int4* x, int t)
{
	return ((int*)x)[t];
}
inline __device__ __host__ int& getInt4(int4* x, int t)
{
	return ((int*)x)[t];
}
inline __device__ __host__ int2& getInt4AsInt2(int4* x, int t)
{
	return ((int2*)x)[t];
}

inline __device__ int LBS(int* a, int L, int R, int x)
{
	while (L < R)
	{
		int mid = (L + R + 1) / 2;
		if (a[mid] <= x)
		{
			L = mid;
		}
		else {
			R = mid - 1;
		}
	}
	return L;
}