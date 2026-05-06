#pragma once
#pragma once
#include <unordered_map>
#include <stack>
class MemCache
{
public:
    void* getMemByAbtrtSize(size_t size);
    void* getMemBySize(size_t size);

    void releaseAbtrtMem(void* ptr, size_t size);
    void releaseMem(void* ptr, size_t size);
    void initAbtrtMem();
    void releaseCache();

    void* getPinnedPage();
    void releasePinnedPage(void* mem);
    void report();
private:
    std::unordered_map<size_t, std::list<void*>> unordered_mp;
    std::vector<std::stack<void*>> stkInVec;
    void* m_pinnedPage{ nullptr };
};

MemCache& getMemCacheRef();