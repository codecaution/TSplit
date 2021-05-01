#include <cmath>
#include <unordered_map>
#include "ps/ps.h"
#include "../common/dlarray.h"
#include <unistd.h>
#include "ps/internal/postoffice.h"
#include "ps/psmodel/PSVector.h"
#include <string>
using namespace ps;

class Worker{
public:

    std::unordered_map<int,int> node2len;
    std::unordered_map<int,PSVector<float>*> node2vec;
    std::unordered_map<int,std::vector<int>> node2timestamp;
    std::unordered_map<int,float*>  node2addr;
    ~Worker() {
      for(auto &ptr: node2vec) {
        delete ptr.second;
      }
    }
    int find_length(int node_name, const DLArray *arr) {
        if(node2len.find(node_name) != node2len.end()) return node2len[node_name];
        int DLArray_len = 1;
        for(int i = 0;i < arr->ndim ;i++) {
            DLArray_len *= arr->shape[i];
        }
        node2len[node_name]=DLArray_len;
        return DLArray_len;
    }
  
    PSVector<float>* find_psvector(int node_name, int len) {
        if (node2vec.find(node_name) != node2vec.end()) return node2vec[node_name];
        std::string name = "weight_" + std::to_string(node_name);
        auto dvec = new PSVector<float>(name,len);
        node2vec[node_name] = dvec;
        return dvec;
    }

    void push(int node_name, const DLArray *arr, bool async = false) {
        int DLArray_len = find_length(node_name, arr);
        float *data= static_cast<float*>(arr->data);
        auto dvec = find_psvector(node_name, DLArray_len);
        if (async == false) {
          dvec->densePush(data, DLArray_len, node2timestamp[node_name], async);
        } else {
          dvec->densePush(data, DLArray_len, node2timestamp[node_name], async); 
        } 
    }

    void pull(int node_name, DLArray *arr, bool async = false) {
        int len = find_length(node_name, arr);
        auto dvec = find_psvector(node_name,len);
        if (async == false) {
          std::vector<float> rets(len);
          dvec->densePull(rets.data(), len, node2timestamp[node_name], async);
          std::copy(rets.begin(), rets.end(), (float*)arr->data);
        } else {
          auto rets = new float[len];
          node2addr[node_name] = rets;
          dvec->densePull(rets, len, node2timestamp[node_name], async); 
        }
    }

    void initAllZeros(int node_name, const DLArray *arr) {
        int len = find_length(node_name, arr);
        auto dvec = find_psvector(node_name, len);
        dvec->initAllZeros();
    }

    void wait_push(int node_name) {
      auto dvec = node2vec[node_name];
      for (auto & t : node2timestamp[node_name])
        dvec->wait(t);
    }

    void wait_pull(int node_name, DLArray *arr) {
      auto dvec = node2vec[node_name];
      for (auto & t : node2timestamp[node_name])
        dvec->wait(t);
      auto rets = node2addr[node_name];
      auto len = node2len[node_name];
      std::copy(rets, rets + len, (float*)arr->data);
      delete rets;
      node2addr[node_name] = nullptr;
    }
};


extern "C" {

void Worker_Init(){
    Start(0);
    if (!IsWorker()) assert(false);
}

Worker worker;

void Worker_Finalize(){
    Finalize(0, true);
}

void Init_All_Zeros(int node_id, const DLArray *arr) {
    worker.initAllZeros(node_id, arr);
}

void DL_Communicate_BSP(int node_id, const DLArray *in_arr, DLArray *out_arr) {
    worker.push(node_id, in_arr);
    Postoffice::Get()->Barrier(0,kWorkerGroup);
    worker.pull(node_id, out_arr);
}

void DL_Communicate_ASP(int node_id, const DLArray *in_arr) {
    worker.push(node_id, in_arr, true);
    worker.pull(node_id, nullptr, true);
}

void  Wait_Push(int node_id) {
    worker.wait_push(node_id);
}

void Wait_Pull(int node_id, DLArray *out_arr) {
    worker.wait_pull(node_id, out_arr);
}

void DL_Communicate_BSP_BY_ASP(int node_id, const DLArray *in_arr, DLArray *out_arr) {
    worker.push(node_id, in_arr, true);
    worker.pull(node_id, nullptr, true);
    Wait_Push(node_id);
    Postoffice::Get()->Barrier(0,kWorkerGroup);
    Wait_Pull(node_id,out_arr);
}

}
