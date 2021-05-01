#include "gpu_memory_manager.h"

int ChunkCounter = 0;
DLMemoryManager *MemoryPool1 = NULL;

#define CUDA_CALL(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

Chunk::Chunk(size_t init_block_size){
    ptr = NULL;
    pre = next = NULL;
    block_size = init_block_size;
    used_size = 0;
    is_used = false;
    bin_id = -1;
    chunk_id = ChunkCounter++;
}
void Chunk::print(){
    cout << endl << "--chunk_id:" << chunk_id << " start--" << endl;
    cout << "Pointer:" << ptr << endl;
    cout << "pre & next:" << pre << " " << next << endl;
    cout << "block_size:" << block_size << " " << used_size << " " << endl;
    cout << "is_used:" << is_used << endl;
    cout << "bin_id:" << bin_id << endl;
    cout << endl << "--chunk_id:" << chunk_id << " end--" << endl;
}

//bool cmp(const ChunkHandle &a, size_t b){
//        return (a -> block_size) < b;
//}

bool Comp::operator ()(const ChunkHandle &a, const ChunkHandle &b) const{
    assert(a != NULL);
    assert(b != NULL);
    return (a -> block_size < b -> block_size || \
        (a -> block_size == b -> block_size && a -> ptr < b -> ptr));
}


Chunk_Bin::Chunk_Bin(){
    bin_id = bin_size = -1;
    contents.clear();
}

void Chunk_Bin::ChunkInsert(ChunkHandle x){
    assert(x != NULL && x -> is_used == false && x -> used_size == 0);
    contents.insert(x);
    x -> bin_id = bin_id;
}

void Chunk_Bin::ChunkDelete(ChunkHandle x)
{
    assert(x != NULL);
    set<ChunkHandle>::iterator it;
    it = contents.find(x);
    if(it == contents.end()){
        printf("KeyError: Chunk_Bin can't find Chunk which try to delete. ");
        assert(false);
    }
    contents.erase(it);
    x -> bin_id = -1;
}

ChunkHandle Chunk_Bin::ChunkSearch(size_t memory_size){
//    ChunkHandle x = new Chunk(memory_size);
//    x -> ptr = 0;
//    cout << memory_size << endl;
//    cout << x -> block_size << endl;
//    set<ChunkHandle>::iterator it = lower_bound(contents.begin(), contents.end(), memory_size, cmp);
    set<ChunkHandle>::iterator it = contents.end();
    for(set<ChunkHandle>::iterator u = contents.begin(); u!=contents.end(); u++ )
        if((*u) -> block_size >= memory_size){
            it = u;
            break;
        }
    // todo: Change find Chunk to binary search
    if(it == contents.end()) return NULL;
    return (*it);
}

void Chunk_Bin::print(){
    cout << endl << "--bin_id:" << bin_id << " start--" << endl;
    cout << "bin_size: " << bin_size << endl;
    cout << "chunks: ";
    for(set<ChunkHandle>::iterator it= contents.begin(); it != contents.end(); it++)
        cout << (*it) -> chunk_id << " ";
    cout << endl << "--bin_id:" << bin_id << " end--" << endl;

}

void* DLMemoryManager::CudaApplyMemory(size_t memory_size){
    void *ret;
    CUDA_CALL(cudaMalloc(&ret, memory_size));
    return ret;
}

void DLMemoryManager::CudeSetDevice(DLContext ctx){
    CUDA_CALL(cudaSetDevice(ctx.device_id));
}

int DLMemoryManager::ChunkHandle2BinNumber(ChunkHandle x){
    int ans = 0;
    long long t = (x -> block_size) / 256;
    assert(x -> block_size != 0);
    while(t >= (1LL << ans))
        ans ++;
    assert(ans != 0);
    return ans - 1;
}

int DLMemoryManager::MemorySize2BinNumber(size_t memory_size){
    //Check here if memory_size is in the middle of two Bins. Maybe the only chunk we need is in Bin_{ans-1}
    size_t ans = 0;
    while((1uLL << ans) * 256 < memory_size) ans++;
    return ans == 0? ans : ans - 1;
}

ChunkHandle DLMemoryManager::DLMemorySearch(size_t memory_size){
    int p = MemorySize2BinNumber(memory_size);
    for(int i = p; i < BINS_NUM; i++){
        ChunkHandle ans = bins[i].ChunkSearch(memory_size);
        if(ans != NULL) return ans;
    }
    return NULL;
}

void DLMemoryManager::DLMemoryInsertIntoBins(ChunkHandle x){
    int pos = ChunkHandle2BinNumber(x);
    bins[pos].ChunkInsert(x);
}

void DLMemoryManager::DLMemoryDeleteFromBins(ChunkHandle x)
{
    int pos = ChunkHandle2BinNumber(x);
    bins[pos].ChunkDelete(x);
}

void DLMemoryManager::DLMemorySplit(ChunkHandle x, size_t memory_size)
{
    size_t t = ceil(memory_size / 256.0) * 256;

    ChunkHandle newx = new Chunk;
    newx -> ptr = /*(float*)*/x -> ptr + t;
    newx -> pre = x;
    newx -> next = x -> next;
    newx -> block_size = (x -> block_size) - t;
    if(newx -> next != NULL)
        (newx -> next) -> pre = newx;
    DLMemoryInsertIntoBins(newx);

    x -> next = newx;
    x -> block_size = t;
}

ChunkHandle DLMemoryManager::DLMemoryChunkMerge(ChunkHandle pre, ChunkHandle now){
    assert(pre -> next == now && now -> pre == pre);
    assert(pre -> is_used == false && now -> is_used == false);

    ChunkHandle newx = new Chunk(pre -> block_size + now -> block_size);
    newx -> ptr = pre -> ptr;
    newx -> pre = pre -> pre;
    newx -> next = now -> next;
    if(pre -> pre != NULL)
        (pre -> pre) -> next = newx;
    if(now -> next != NULL)
        (now -> next) -> pre = newx;
    if(pre -> bin_id != -1)
        DLMemoryDeleteFromBins(pre);
    if(now -> bin_id != -1)
        DLMemoryDeleteFromBins(now);
    delete pre;
    delete now;
//    cout << "Chunk Merge end" << endl;
    return newx;
}

bool DLMemoryManager::DLMemoryTry(size_t memory_size)
{
// todo: this is only a test

//    CUDA_CALL(cudaSetDevice(0));
//    void *ret;
//    cudaError_t e = cudaMalloc(&ret, memory_size);
//    if( e != cudaSuccess ) return false;
//    else{
//        CUDA_CALL(cudaFree(ret));
//        return true;
//    }
    ChunkHandle p = DLMemorySearch(memory_size);
    if(p == NULL)
        return false;
    return true;
}

void* DLMemoryManager::DLMemoryMalloc(size_t memory_size){
//    //todo: this is only a test
//    CUDA_CALL(cudaSetDevice(0));
//    void *ret;
//    CUDA_CALL(cudaMalloc(&ret, memory_size));
//    if(count_free_time.count(ret)==0) count_free_time[ret]=0;
//    count_free_time[ret]++;
//    // cout << "Malloc " << ret << " " << memory_size << endl;
//    return ret;
//    cout << "malloc" << " " << memory_size << endl;
    ChunkHandle p = DLMemorySearch(memory_size);
    if(p == NULL){
        printf("Out of Memory!");
        assert(1 == -1);
    }
    Address2ChunkHandle[p -> ptr] = p;
//    Address2Size[p -> ptr] = memory_size;
    p -> is_used = true;
    p -> used_size = memory_size;
    DLMemoryDeleteFromBins(p);

    if(p -> block_size > 2 * ceil(memory_size / 256.0) * 256 || (p -> block_size - memory_size >= (128uLL << 20)))
        DLMemorySplit(p, memory_size);

    return p -> ptr;
}

DLMemoryManager::DLMemoryManager(DLContext ctx, size_t GpuMemoryCapacity){
////  todo: this is only a test
//      void* ptr;
    CudeSetDevice(ctx);
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    GpuMemoryCapacity = avail - 500 * 1024 * 1024;
    // std::cout<<"GPU Memory="<<GpuMemoryCapacity<<std::endl;
    for(size_t i = 0; i < BINS_NUM; i ++){
        bins[i].bin_id = i;
        bins[i].bin_size = 256 * (1LL << i);
    }
    void* ptr = CudaApplyMemory(GpuMemoryCapacity);
    if(ptr == NULL){
        printf("DLMemoryManager apply for initial memory of gpu false. ");
        assert(1 == -1);
    }
    ptr_id = 0;
    //head node
    chunk_list = new Chunk(0);
    chunk_list -> is_used = true;

    chunk_list -> next = new Chunk(GpuMemoryCapacity/256 * 256);
    (chunk_list -> next) -> pre = chunk_list;
    (chunk_list -> next) -> ptr = ptr;

    DLMemoryInsertIntoBins(chunk_list -> next);
}

void DLMemoryManager::DLMemoryFree(void* ptr)
{
////todo: this is only a test for cudaFree
//    CUDA_CALL(cudaSetDevice(0));
//    void* ret = ptr;
//    if(count_free_time.count(ret)==0 || count_free_time[ret]==0){
//        cout << "gg " << (count_free_time.count(ret)==0 ) << " " << (count_free_time[ret]==0) << endl;
//    }
//    count_free_time[ret]--;
//    if(count_free_time[ret]==0){
//        count_free_time.erase(ret);
//    }
//    CUDA_CALL(cudaFree(ptr));
//    check here if a pointer is free twice
//    cout << "Free" << " " << Address2Size[ptr] << endl;
    assert(Address2ChunkHandle.count(ptr) != 0);
    ChunkHandle p = Address2ChunkHandle[ptr];
    assert(p != NULL);
    Address2ChunkHandle.erase(ptr);
    p -> is_used = false;
    p -> used_size = 0;
    ChunkHandle pre = p -> pre, next = p -> next;
    if(pre != NULL && pre -> is_used == false)
        p = DLMemoryChunkMerge(pre, p);
    if(next != NULL && next -> is_used == false)
        p = DLMemoryChunkMerge(p, next);
    DLMemoryInsertIntoBins(p);
}

int DLMemoryManager::DLMemoryPrintState(int type)
{
    if(type) printf("------------DLMemoryManger State------------\n");
    ChunkHandle p = chunk_list -> next;
    int t = 0;
    while(p != NULL){
        if(p -> is_used == false) t++;
        if(type) p -> print();
        p = p -> next;
    }
    if(type){
        printf("\n\n-------------------\n\n");
        for(int i = 0; i < BINS_NUM; i++)
            bins[i].print();
        printf("----------DLMemoryManger State End----------\n");
    }
    return t;
}

void DLMemoryManager::DLMemoryUsage(size_t &used_size, size_t &unused_unavailable_size, size_t &unused_available_size)
{
    used_size = unused_available_size = unused_unavailable_size = 0;
    ChunkHandle p = chunk_list -> next;
    while(p != NULL){
        if(p -> is_used){
            used_size += p -> used_size;
            unused_unavailable_size += (p -> block_size - p -> used_size);
        }
        else
            unused_available_size += p -> block_size;
        p = p -> next;
    }
}


