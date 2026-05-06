#pragma once
typename Point2d;
struct int3;
double getBitmap(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int*& bitmapOffsets, int*& bitmapTris, int numTriangles);