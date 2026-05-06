#include "utils.cuh"
#include <cub/cub.cuh>
#include "math_base.cuh"

struct CustomLess
{
	__device__ inline bool operator()(const int3& a, const int3& b)
	{
		return a.x < b.x || (a.x == b.x && a.y < b.y);
	}

	__device__ bool operator()(const int2& lhs, const int2& rhs)
	{
		return (lhs.x < rhs.x) || (lhs.x == rhs.x && lhs.y < rhs.y);
	}
};

template<typename FUNC, typename... ARGS>
void utils::kernel(FUNC func, unsigned int a, unsigned int b, ARGS... args) {
	if (a == 0) return;
	func << <(a + b - 1) / b, b >> > (args...);
}

template<typename FUNC, typename... ARGS>
float utils::kernel(const std::string& name, FUNC func, unsigned int a, unsigned int b, ARGS... args) {
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	func << <(a + b - 1) / b, b >> > (args...);
	//getLastCudaError(name.c_str());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << name << " time: " << elapsedTime << " ms\n";
	//printf("%s time: %f ms\n", name.c_str(), elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime;
}

template<typename T>
void utils::exlusiveScanRegist(T* d_in, T* d_out, int num_items, size_t& cubTempStorageBytes)
{
	////!!!error in cub::DeviceSccan::ExlusiveSum. idk why.
	//cub::DeviceScan::ExclusiveSum(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceScan::ExclusiveSum(storage, storage_bytes, d_in, d_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;

	//size_t part_size = 1024;  // tuned
	//size_t part_num = ((num_items + part_size - 1) / part_size) * sizeof(T) * 64;
	//if (part_num > cubTempStorageBytes) cubTempStorageBytes = part_num;
}

template<typename T>
__device__ T ScanWarp(T val) {
	int32_t lane = threadIdx.x & 31;
	T tmp = __shfl_up_sync(0xffffffff, val, 1);
	if (lane >= 1) {
		val += tmp;
	}
	tmp = __shfl_up_sync(0xffffffff, val, 2);
	if (lane >= 2) {
		val += tmp;
	}
	tmp = __shfl_up_sync(0xffffffff, val, 4);
	if (lane >= 4) {
		val += tmp;
	}
	tmp = __shfl_up_sync(0xffffffff, val, 8);
	if (lane >= 8) {
		val += tmp;
	}
	tmp = __shfl_up_sync(0xffffffff, val, 16);
	if (lane >= 16) {
		val += tmp;
	}
	return val;
}

template<typename T>
__device__ __forceinline__ T ScanBlock(T val) {
	int32_t warp_id = threadIdx.x >> 5;
	int32_t lane = threadIdx.x & 31;
	extern __shared__ T warp_sum[];
	// scan each warp
	val = ScanWarp(val);
	__syncthreads();
	// write sum of each warp to warp_sum
	if (lane == 31) {
		warp_sum[warp_id] = val;
	}
	__syncthreads();
	// use a single warp to scan warp_sum
	if (warp_id == 0) {
		warp_sum[lane] = ScanWarp(warp_sum[lane]);
	}
	__syncthreads();
	// add base
	if (warp_id > 0) {
		val += warp_sum[warp_id - 1];
	}
	__syncthreads();
	return val;
}

template<typename T>
__global__ void ReducePartSumKernelSinglePass(const T* input,
	T* g_part_sum, size_t n,
	size_t part_size) {
	// this block process input[part_begin:part_end]
	size_t part_begin = blockIdx.x * part_size;
	size_t part_end = min((blockIdx.x + 1) * part_size, n);
	// part_sum
	int32_t part_sum = 0;
	for (size_t i = part_begin + threadIdx.x; i < part_end; i += blockDim.x) {
		part_sum += input[i];
	}
	using BlockReduce = cub::BlockReduce<T, 1024>;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	part_sum = BlockReduce(temp_storage).Sum(part_sum);
	__syncthreads();
	if (threadIdx.x == 0) {
		g_part_sum[blockIdx.x] = part_sum;
	}
}
template<typename T>
__global__ void ScanWithBaseSumSinglePass(const T* input,
	T* g_base_sum, T* output,
	size_t n, size_t part_size,
	bool debug) {
	// base sum
	__shared__ T base_sum;
	if (threadIdx.x == 0) {
		if (blockIdx.x == 0) {
			base_sum = 0;
		}
		else {
			base_sum = g_base_sum[blockIdx.x - 1];
		}
	}
	__syncthreads();
	// this block process input[part_begin:part_end]
	size_t part_begin = blockIdx.x * part_size;
	size_t part_end = (blockIdx.x + 1) * part_size;
	for (size_t i = part_begin + threadIdx.x; i < part_end; i += blockDim.x) {
		T val = i < n ? input[i] : 0;
		T backup = val;
		val = ScanBlock(val);
		if (i < n) {
			if (!debug) {
				output[i] = val + base_sum - backup;
			}
			else {
				output[i] = val + base_sum;
			}
		}
		__syncthreads();
		if (threadIdx.x == blockDim.x - 1) {
			base_sum += val;
		}
		__syncthreads();
	}
}
template<typename T>
void utils::exlusiveScan(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items)
{
	//!!!error in cub::DeviceSccan::ExlusiveSum. idk why.
	cub::DeviceScan::ExclusiveSum(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);
	//size_t part_num = 1024;
	//size_t part_size = (num_items + part_num - 1) / part_num;
	//T* part_sum = (T*)cubTempStorage;
	//ReducePartSumKernelSinglePass << <part_num, 1024 >> > (d_in, part_sum, num_items,
	//	part_size);
	//ScanWithBaseSumSinglePass << <1, 1024, 32 * sizeof(T) >> > (
	//	part_sum, (T*)nullptr, part_sum, part_num, part_num, true);
	//ScanWithBaseSumSinglePass << <part_num, 1024, 32 * sizeof(T) >> > (
	//	d_in, part_sum, d_out, num_items, part_size, false);
}

template<typename T>
void utils::sortRegist(T* d_in, T* d_out, int* label, int* label_out, int num_items, size_t& cubTempStorageBytes)
{
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(storage, storage_bytes, d_in, d_out, label, label_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::sortRegist(T* d_in, T* d_out, int3* label, int3* label_out, int num_items, size_t& cubTempStorageBytes)
{
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(storage, storage_bytes, d_in, d_out, label, label_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::sort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int3* label, int3* label_out, int num_items)
{
	cub::DeviceRadixSort::SortPairs(cubTempStorage, cubTempStorageBytes, d_in, d_out, label, label_out, num_items);
	getLastCudaError("sort");
}

template<typename T>
void utils::sort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int2* label, int2* label_out, int num_items)
{
	cub::DeviceRadixSort::SortPairs(cubTempStorage, cubTempStorageBytes, d_in, d_out, label, label_out, num_items);
	getLastCudaError("sort");
}

template<typename T>
void utils::sort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int* label, int* label_out, int num_items)
{
	cub::DeviceRadixSort::SortPairs(cubTempStorage, cubTempStorageBytes, d_in, d_out, label, label_out, num_items);
	getLastCudaError("sort");
}

template<typename T>
void utils::mergeSortRegist(T* key, int* value, int num_items, size_t& cubTempStorageBytes)
{
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceMergeSort::SortPairs(storage, storage_bytes, key, value, num_items, CustomLess());
	getLastCudaError("merge sort regist");
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::mergeSort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, int* d_out, int num_items)
{
	cub::DeviceMergeSort::SortPairs(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items, CustomLess());
	getLastCudaError("sort");
}

template<typename T>
void utils::minReduceRegist(int num_items, size_t& cubTempStorageBytes) {
	T* d_in = nullptr;
	T* d_out = nullptr;
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceReduce::Min(storage, storage_bytes, d_in, d_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::maxReduceRegist(int num_items, size_t& cubTempStorageBytes) {
	T* d_in = nullptr;
	T* d_out = nullptr;
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceReduce::Max(storage, storage_bytes, d_in, d_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::minReduce(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items)
{
	cub::DeviceReduce::Min(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);
}

template<typename T>
void utils::maxReduce(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items)
{
	cub::DeviceReduce::Max(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);
}

template<typename T>
float utils::minReduce(const std::string& name, void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	printf("reduce %d\n", num_items);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cub::DeviceReduce::Min(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%s time: %f ms\n", "min reduce", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime;
}

template<typename T>
float utils::maxReduce(const std::string& name, void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	printf("reduce %d\n", num_items);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cub::DeviceReduce::Max(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%s time: %f ms\n", "min reduce", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime;
}

#define RKERNEL(__args__, ...) template void utils::kernel<>(void(*)(__args__, __VA_ARGS__), unsigned int, unsigned int,__args__, __VA_ARGS__); \
                               template float utils::kernel<>(const std::string&, void(*)(__args__, __VA_ARGS__), unsigned int, unsigned int,__args__, __VA_ARGS__);

template void utils::minReduce<float>(void* cubTempStorage, size_t cubTempStorageBytes, float*, float*, int);
template void utils::maxReduce<float>(void* cubTempStorage, size_t cubTempStorageBytes, float*, float*, int);
template float utils::minReduce<float>(const std::string&, void* cubTempStorage, size_t cubTempStorageBytes, float*, float*, int);
template float utils::maxReduce<float>(const std::string&, void* cubTempStorage, size_t cubTempStorageBytes, float*, float*, int);
template void utils::minReduceRegist<float>(int, size_t&);
template void utils::sortRegist<unsigned int>(unsigned int*, unsigned int*, int*, int*, int, size_t&);
template void utils::maxReduceRegist<float>(int, size_t&);
template void utils::sort<unsigned int>(void* cubTempStorage, size_t cubTempStorageBytes, unsigned int*, unsigned int*, int*, int*, int);
template void utils::sort<int>(void* cubTempStorage, size_t cubTempStorageBytes, int*, int*, int*, int*, int);
template void utils::exlusiveScanRegist<int>(int*, int*, int, size_t&);
template void utils::exlusiveScan<int>(void*, size_t, int*, int*, int);
template void utils::mergeSortRegist<int2>(int2*, int*, int num_items, size_t& cubTempStorageBytes);
template void utils::mergeSort<int2>(void*, size_t, int2* d_in, int* d_out, int num_items);

template void utils::sortRegist<int>(int* d_in, int* d_out, int3* label, int3* label_out, int num_items, size_t& cubTempStorageBytes);
template void utils::sort<int>(void* cubTempStorage, size_t cubTempStorageBytes, int* d_in, int* d_out, int3* label, int3* label_out, int num_items);
template void utils::sort<int>(void* cubTempStorage, size_t cubTempStorageBytes, int* d_in, int* d_out, int2* label, int2* label_out, int num_items);
RKERNEL(Point2d*, double*, long long, int) //transPoints
RKERNEL(Point2d*, int*, int3*, int3*, PointInteger, int) //insertBigTriangle
RKERNEL(Point2d*, int*, int4*, PointInteger, int) //initTriangle
RKERNEL(int*, int4*, int*, int*, int) //choosePoint
RKERNEL(Point2d*, int3*, int*, int4*, int*, int2*, int) //choosePointSmart_s1
RKERNEL(int*, int4*, int*, int2*, int) //choosePointSmart_s2

RKERNEL(int*, int*, int*, int) //addNewTriangle_step1 //samplePoints_s2
RKERNEL(int*, int3*, int3*, int*, int*, int*, int4*, int) //addNewTriangle_step2
RKERNEL(int*, int3*, int*, int*, int*, int4*, int) //addNewTriangle_step2_simple
RKERNEL(Point2d*, int*, int3*, int*, int*, int4*, int) //scatterPoints

RKERNEL(int*, int4*, int*, int*, int, int) //sortPoints_s1

RKERNEL(Point2d*, int*, int*, unsigned int, unsigned int, int) //samplePoints_s1
RKERNEL(Point2d*, Point2d*, int*, int) //samplePoints_s3
RKERNEL(Point2d*, int3*, int*, unsigned int, unsigned int, int) //countTriAABB
RKERNEL(Point2d*, int3*, int*, int*, int*, unsigned int, unsigned int, int, int) //intersectSquareTri_s1
RKERNEL(Point2d*, int3*, int*, int*, int*, int*, unsigned int, unsigned int, int, int) //intersectSquareTri_s2

RKERNEL(int3*, int*, int, int, int) //convertSample
RKERNEL(Point2d*, int3*, int*, int4*, int*, int*, int) //initTriangleBoost
RKERNEL(Point2d*, int*, PointInteger, int) //insertCornerPoints

RKERNEL(int3*, int3*, int*, int*, int*, int4*, int) //updateOnEdge
RKERNEL(Point2d*, int3*, int3*, int*, int) //updateAdjTris_new
RKERNEL(int3*, int3*, int*, int) //testAdjTris  //updateAdjTris

RKERNEL(int*, int*, int) //filterTriangles_s1 //compactFlip
RKERNEL(int3*, int3*, int*, int*, int) //filterTriangles_s2
RKERNEL(int3*, int3*, int3*, int3*, int*, int*, int) //filterTriangles_s2_new

RKERNEL(int3*, int3*, int*, int*, int*, int)
RKERNEL(Point2d*, int2*, unsigned long long*, unsigned long long*, int, int) //locatePointsLB_s1
RKERNEL(unsigned long long*, int*, unsigned long long*, int, int) //locatePointsLB_s2
RKERNEL(Point2d*, int3*, int*, int2*, int*, int*, int2*, int*, PointInteger, int, int) //locatePointsLB_s3
RKERNEL(Point2d*, int3*, int*, int2*, int*, int*, int*, int*, int2*, int*, int, int) //locatePointsLB_s3_boost
RKERNEL(Point2d*, int3*, int3*, int2*, int*, int*, int*, int) //locateStart

RKERNEL(Point2d*, int3*, int3*, int2*, int2*, int*, int*, int*, int*, int) //overlapTriangleCount
RKERNEL(Point2d*, int3*, int3*, int2*, int2*, int*, int*, int*, int*, int*, int*, int*, int*, int) //overlapTriangle

RKERNEL(int*, int*, int*, int*, int*, int) //findIntersectedCons
RKERNEL(Point2d*, int3*, int2*, int*, int*, int*, int3*, int3*, int) //reduceCons

RKERNEL(int3*, int2*, int*, int*, int3*, int3*, int) //formPolygon
RKERNEL(Point2d*, int3*, int2*, int*, int*, int3*, int3*, int*, int) //formPolygon_s1
RKERNEL(int3*, int2*, int*, int*, int*, int*, int*, int) //formPolygon_s2
RKERNEL(Point2d*, int2*, int*, int*, MyInteger<logPointRes*2+1>*, int) //calDistance
RKERNEL(MyInteger<logPointRes * 2 + 1>*, int*, int*, int) //makeTree
RKERNEL(int2*, MyInteger<logPointRes * 2 + 1>*, int*, int*, int3*, int) //reTriangle

RKERNEL(int2*, int*, int*, int3*, int*, int*, int*, int*, int)
RKERNEL(int2*, int3*, int2*, int*, int*, int*, int*, int2*, int*, int)
RKERNEL(int2*, int*, int2*, int*, int*, int*, int)
RKERNEL(int3*, int)
RKERNEL(int3*, int3*, int*, int3*, int)
RKERNEL(int3*, int3*, int*, int3*, int*, int)
RKERNEL(unsigned int*, int)
RKERNEL(Point2d*, int3*, int3*, int*, int*, int*, unsigned int*, int*, int) //testInCircle_s1
RKERNEL(int3*, int*, int*, int*, unsigned int*, int) //testInCircle_s2
RKERNEL(int3*, int3*, int*, int*, int) //flipEdge
RKERNEL(int3*, int3*, int) //flipEdge_heal

RKERNEL(Point2d*, int3*, int2*, int*, int*, int*, unsigned int*, int*, int) //testInCircle_s1_e
RKERNEL(int2*, int*, int*, int*, int*, unsigned int*, int) //testInCircle_s2_e
RKERNEL(int2*, int2*, int3*, int*, int) //flipEdge_e
RKERNEL(int3*, int2*, int2*, int*, int*, int) //flipEdge_heal_e
RKERNEL(Point2d*, int3*, int2*, int2*, int*, int*, int*, int*, int) //flipEdge_healAndtest
RKERNEL(int3*, int2*, int2*, int*, int*, int*, int) //flipEdge_testAndflip
RKERNEL(Point2d*, int3*, int2*, int*, int*, int*, int) //initNeedFlip

RKERNEL(int*, int) //clearNeedTest
RKERNEL(Point2d*, int3*, int2*, int*, int*, int*, int*, int*, int, int) //testInCircle_s1_e_c

RKERNEL(int3*, int2*, int*, int) //remakeAdjFaceSort_s1
RKERNEL(int2*, int*, int3*, int) //remakeAdjFaceSort_s2

RKERNEL(int3*, int*, int3*, int) //remakeAdjFaceRadixSort_s1
RKERNEL(int*, int3*, int3*, int) //remakeAdjFaceRadixSort_s2

RKERNEL(int3*, int*, int*, int) //remakeEdgeRadixSort_s1_new
RKERNEL(int2*, int*, int*, int, int) //remakeEdgeRadixSort_s1_addC
RKERNEL(int*, int3*, int*, int) //remakeEdgeRadixSort_s2
RKERNEL(int3*, int*, int*, int*, int) //remakeEdgeRadixSort_s2_new
RKERNEL(int*, int3*, int*, int2*, int2*, int) //remakeEdgeRadixSort_s3
RKERNEL(int3*, int*, int*, int*, int2*, int2*, int*, int) //remakeEdgeRadixSort_s3_new

RKERNEL(Point2d*, int2*, int*, int*, unsigned int, int) //reorderEdge_s1
RKERNEL(int2*, int2*, int2*, int2*, int*, int) //reorderEdge_s2

RKERNEL(Point2d*, int3*, int*, int*, unsigned int, int) //reorderFace_s1
RKERNEL(int3*, int3*, int*, int) //reorderFace_s2

RKERNEL(Point2d*, int*, int*, unsigned int, int) //reorderPoint_s1
RKERNEL(Point2d*, Point2d*, int*, int*, int) //reorderPoint_s2
RKERNEL(int3*, int*, int) //reorderPoint_s3 //excludeSamples
RKERNEL(int2*, int*, int) //reorderPoint_s4

RKERNEL(int3*, int2*, int2*, int2*, int) //rebuildEdgeInfo
RKERNEL(int3*, int2*, int2*, int2*, int*, int) //rebuildFaceInfo
RKERNEL(int3*, int3*, int2*, int2*, int) //rebuildFaceAdjInfo

RKERNEL(Point2d*, int2*, int2*, int2*, int*, int*, int*, int) //initNeedFlip_n
RKERNEL(Point2d*, int2*, int2*, int2*, int*, int*, int*, int*, int) //flipEdge_healAndtest_n
RKERNEL(int2*, int2*, int2*, int*, int*, int*, int) //flipEdge_testAndflip_n

RKERNEL(int3*, int2*, int2*, int3*, int) //getFace2Edge
RKERNEL(int3*, int2*, int2*, int2*, int4*, int) //getEdge2Edge

RKERNEL(Point2d*, int2*, int4*, int*, int) //testDelaunay
RKERNEL(Point2d*, int2*, Point2d2*, int) //makeEdgeWithPoint
RKERNEL(Point2d*, Point2d2*, int2*, int4*, int*, int*, int*, int*, int) //initNeedFlip_new
RKERNEL(int2*, int*, int*, int*, int*, int) //confirmFlip
RKERNEL(Point2d*, int2*, Point2d2*, int4*, int*, int*, int*, int*, int) //multiFlip
RKERNEL(Point2d*, int2*, int2*, int4*, int*, int*, int*, int*, int*, int) //multiFlipWithStack
RKERNEL(Point2d*, int2*, int2*, int2*, int4*, int*, int*, int*, int*, int) //multiFlipSide
RKERNEL(int3*, int2*, int2*, int3*, int) //updateFace2EdgeAndEdge2Face
RKERNEL(Point2d*, int2*, int2*, int3*, int3*, int*, int*, int*, int*, int*, int) //initNeedFlipFE_new
RKERNEL(Point2d*, int2*, int3*, int3*, int*, int*, int*, int*, int*, int) //multiFlipFE

RKERNEL(int3*, int*, int*, int) //countEdges
RKERNEL(int3*, int3*, int*, int*, int) //specialPoints
RKERNEL(int3*, int3*, int3*, int3*, int*, int*, int*, int*, int) //formEdges
RKERNEL(int3*, int3*, int3*, int3*, int*, int) //formOthers
RKERNEL(Point2d*, int*, Point2d*, int) //makePointList
RKERNEL(Point2d*, Point2d*, int*, int*, int*, int*, int) //testNeighborInCircle
RKERNEL(Point2d*, Point2d*, int*, int*, int*, int*, int) //starSplay