#pragma once
#include <cuda.h>

#include "cuda_runtime.h"
#include "cudart_platform.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "MemCache.h"
//#include "device/device_scan.cuh"

#include <iostream>
class utils {
public:
	template<typename FUNC, typename... ARGS>
	static void kernel(FUNC func, unsigned int a, unsigned int b, ARGS... args);

	template<typename FUNC, typename... ARGS>
	static float kernel(const std::string& name, FUNC func, unsigned int a, unsigned int b, ARGS... args);

	template<typename T>
	static inline void memcpy(T* dst, const T* src, int num, cudaMemcpyKind kind = cudaMemcpyHostToDevice, int offset = 0)
	{
		checkCudaErrors(cudaMemcpy(dst + offset, src, num * sizeof(T), kind));
	}

	template<typename T>
	static inline void mallocAndCpy(T*& dst, const T* src, int num)
	{
		utils::malloc(dst, num);
		utils::memcpy(dst, src, num);
	}
	template<typename T>
	static inline void release(T*& ptr, size_t num)
	{
		if (ptr != nullptr) {
			getMemCacheRef().releaseAbtrtMem((void*)ptr, num * sizeof(T));
			ptr = nullptr;
		}
	}
	template<typename... T, typename T1>
	static inline void free(T1*& argvs1, T*&... argvs)
	{
		if (argvs1 != nullptr) {
			cudaFree(argvs1);
			argvs1 = nullptr;
		}
		free(argvs...);
	}

	template<typename T>
	static inline void free(T*& p)
	{
		if (p != nullptr) {
			cudaFree(p);
			p = nullptr;
		}
	}

	template<typename T>
	static inline void memset(T* mem, int nn, int value = 0) {
		checkCudaErrors(cudaMemset(mem, value, nn * sizeof(T)));
	}

	template<typename T>
	static inline void malloc(T*& mem, int nn) {
		mem = (T*)getMemCacheRef().getMemByAbtrtSize(nn * sizeof(T));
		//checkCudaErrors(cudaMalloc((void**)&mem, nn * sizeof(T)));
	}

	template<typename T>
	static inline void getValue(T* ret, T* mem, int N) {
		checkCudaErrors(cudaMemcpy(ret, mem + N, sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	static inline T getValue(T* mem, int N) {
		T ret;
		checkCudaErrors(cudaMemcpy(&ret, mem + N, sizeof(T), cudaMemcpyDeviceToHost));
		return ret;
	}

	template<typename T>
	static void exlusiveScanRegist(T* d_int, T* d_out, int num_items, size_t& cubTempStorageBytes);

	template<typename T>
	static void exlusiveScan(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items);

	template<typename T>
	static void sortRegist(T* d_in, T* d_out, int* label, int* label_out, int num_items, size_t& cubTempStorageBytes);

	template<typename T>
	static void sortRegist(T* d_in, T* d_out, int3* label, int3* label_out, int num_items, size_t& cubTempStorageBytes);

	template<typename T>
	static void sort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int* label, int* label_out, int num_items);

	template<typename T>
	static void sort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int3* label, int3* label_out, int num_items);

	template<typename T>
	static void sort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int2* label, int2* label_out, int num_items);

	template<typename T>
	static void mergeSortRegist(T* d_key, int* d_value, int num_items, size_t& cubTempStorageBytes);

	template<typename T>
	static void mergeSort(void* cubTempStorage, size_t cubTempStorageBytes, T* d_key, int* d_value, int num_items);

	template<typename T>
	static void minReduceRegist(int num_items, size_t& cubTempStorageBytes);

	template<typename T>
	static void maxReduceRegist(int num_items, size_t& cubTempStorageBytes);

	template<typename T>
	static void minReduce(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items);

	template<typename T>
	static void maxReduce(void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items);

	template<typename T>
	static float maxReduce(const std::string& name, void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items);

	template<typename T>
	static float minReduce(const std::string& name, void* cubTempStorage, size_t cubTempStorageBytes, T* d_in, T* d_out, int num_items);
};
