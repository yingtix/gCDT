#pragma once
typename Point2d;
struct int3;
struct int2;
double addConstraints(void*, size_t, Point2d* points, int3* tris, int3* adjTris, int* sons, int*, int*, int2* cons, int3*&, int numCons, int numPoints, int numTriangles, int&);
double addConstraints(void*, size_t, Point2d* points, int3* tris, int3* adjTris, int* sons, int2* cons, int numCons, int numPoints, int numTriangles);