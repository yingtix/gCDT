#pragma once
typename Point2d;
struct int3;
double rawMeshBefore(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int* sons, int& numTriangles, int numPoints);
double rawMeshAfter(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int* bitmapOffsets, int* bitmapTriIds, int& numTriangles, int numPoints);
double getSamplePoints(void*, size_t, Point2d* points, int numPoints, Point2d*& samplePoints, int*& sample, int& num);
void rawMesh(void* cubTempStorage, size_t cubTempStorageBytes, Point2d* points, int3* tris, int3* adjTris, int* sons, int& numTriangles, int numPoints);