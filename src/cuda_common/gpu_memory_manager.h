#include "dlarray.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

typedef class Chunk{
public:
    void *ptr; //base address
    Chunk *pre,*next;
    size_t block_size, used_size;
    bool is_used;
    int bin_id;
    int chunk_id;

    Chunk(size_t init_block_size = 0);
    void print();
}Chunk;

typedef Chunk* ChunkHandle;

class Comp{
public:
    bool operator ()(const ChunkHandle &a, const ChunkHandle &b) const;
};

class Chunk_Bin
{
public:

    int bin_id;
    size_t bin_size;
    set<ChunkHandle, Comp> contents;


    Chunk_Bin();
    void ChunkInsert(ChunkHandle x);
    void ChunkDelete(ChunkHandle x);
    ChunkHandle ChunkSearch(size_t memory_size);
    void print();
};

class DLMemoryManager{
public:
    //unfinished
    DLMemoryManager(DLContext ctx, size_t GpuMemoryCapacity = 22LL * 1024 * 1024 * 1024);
    void* DLMemoryMalloc(size_t memory_size);
    void DLMemoryFree(void* ptr);
    int DLMemoryPrintState(int type);
    void DLMemoryUsage(size_t &used_size, size_t &unused_unavailable_size, size_t &unused_available_size);
    bool DLMemoryTry(size_t memory_size);

private:
    static const int BINS_NUM = 30;
    Chunk_Bin bins[BINS_NUM];
    ChunkHandle chunk_list;
    int ptr_id;
    map<void*, ChunkHandle> Address2ChunkHandle;

    void CudeSetDevice(DLContext ctx);
    void* CudaApplyMemory(size_t memory_size);
    int ChunkHandle2BinNumber(ChunkHandle x);
    int MemorySize2BinNumber(size_t memory_size);
    ChunkHandle DLMemorySearch(size_t memory_size);
    ChunkHandle DLMemoryChunkMerge(ChunkHandle pre, ChunkHandle now);
    void DLMemorySplit(ChunkHandle x, size_t memory_size);
    void DLMemoryInsertIntoBins(ChunkHandle x);
    void DLMemoryDeleteFromBins(ChunkHandle x);
};