#pragma
#include "memCache.h"
#include <cuda_runtime.h>

static const std::vector<size_t> sizes{ 32, 1024, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 25165824, 536870912 };
static const std::vector<size_t> sizesInitNum{ 32, 32, 16, 16, 16, 16, 16, 8, 8, 4, 1, 6};
void* MemCache::getMemByAbtrtSize(size_t size)
{
    void* memAddress = nullptr;
    auto sz = std::lower_bound(sizes.begin(), sizes.end(), size);
    if (sz == sizes.end())
    {
        printf("large Mem: %lld\n", size);
        cudaMalloc(&memAddress, size);
        return memAddress;
    }
    else {
        size_t ad = sz - sizes.begin();
        if (stkInVec[ad].empty())
        {
            printf("new Mem: %lld\n", size);
            cudaMalloc(&memAddress, sizes[ad]);
            stkInVec[ad].push(memAddress);
        }

        memAddress = stkInVec[ad].top();
        stkInVec[ad].pop();
        return memAddress;
    }
}

void* MemCache::getMemBySize(size_t size)
{
    if (unordered_mp.find(size) != unordered_mp.end())
    {
        if (unordered_mp[size].empty())
        {
            return nullptr;
        }
        else
        {
            auto back = unordered_mp[size].back();
            unordered_mp[size].pop_back();
            return back;
        }
    }
    return nullptr;
}

void* MemCache::getPinnedPage()
{
    return m_pinnedPage;
}

void MemCache::releasePinnedPage(void* mem)
{
    m_pinnedPage = mem;
}

void MemCache::initAbtrtMem()
{
    void* memAddress = nullptr;
    for (int i = 0; i < sizes.size(); i++)
    {
        std::stack<void*> stk;
        stkInVec.push_back(std::move(stk));
    }
    for (int i = 0; i < sizes.size(); i++)
    {
        for (int j = 0; j < sizesInitNum[i]; j++)
        {
            cudaMalloc(&memAddress, sizes[i]);
            stkInVec[i].push(memAddress);
        }
    }
}

void MemCache::report()
{
    printf("pool report\n");
    for (int i = 0; i < sizes.size(); i++)
    {
        printf("%d: %d\n", i, stkInVec[i].size());
    }
}

void MemCache::releaseAbtrtMem(void* ptr, size_t size)
{
    void* memAddress = nullptr;
    auto sz = std::lower_bound(sizes.begin(), sizes.end(), size);
    if (sz == sizes.end())
    {
        cudaFree(ptr);
    }
    else {
        size_t ad = sz - sizes.begin();
        stkInVec[ad].push(ptr);
    }
}

void MemCache::releaseMem(void* ptr, size_t size)
{
    unordered_mp[size].push_back(ptr);
}

MemCache& getMemCacheRef()
{
    static MemCache cache;
    static bool first = true;
    if (first)
    {
        first = false;
        cache.initAbtrtMem();
    }
    return cache;
}

void MemCache::releaseCache()
{
    for (auto& item : unordered_mp)
    {
        for (auto ptr : item.second)
        {
            cudaFree(ptr);
        }
    }
    cudaFreeHost(m_pinnedPage);
    m_pinnedPage = nullptr;
}