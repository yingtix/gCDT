#pragma once

typename Point2d;
struct int3;
double tryFastDelaunay(void* cubTemp, size_t cubBytes, Point2d* points, int3* tris, int3* adjTris, int numPoints, int numTriangles);