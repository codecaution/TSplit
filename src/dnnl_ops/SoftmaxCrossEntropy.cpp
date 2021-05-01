//#include "../dnnl_test.h"
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <omp.h>
#include <sys/time.h>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"
using namespace dnnl;

//DLArrayHandle A,B,C,D;

extern "C" int DnnlSoftmaxCrossEntropy(const DLArrayHandle A,const DLArrayHandle B, DLArrayHandle output)
{
     //engine eng(engine::kind::cpu, 0);
    //stream engine_stream(eng);
      dnnl_stream_init();
  // TODO: Your code here


    assert(A -> ndim == 2 && B -> ndim == 2 && A -> shape[0] == B -> shape[0] && A -> shape[1] == B -> shape[1] );
    float *C = new float[(A -> shape[0]) * (A -> shape[1])];
    auto mat_md = memory::desc({A -> shape[0], A-> shape[1]}, memory::data_type::f32, memory::format_tag::ab);
    auto src_mem_1 = memory(mat_md, eng, A -> data);
    auto src_mem_2 = memory(mat_md, eng, B -> data);
    auto dst_logsoftmax_mem = memory(mat_md, eng, C);

    auto Logsoftmax_d = logsoftmax_forward::desc(prop_kind::forward_training, mat_md, 1);
    auto Logsoftmax_pd = logsoftmax_forward::primitive_desc(Logsoftmax_d, eng);
    auto Logsoftmax = logsoftmax_forward(Logsoftmax_pd);

    Logsoftmax.execute(engine_stream, {{DNNL_ARG_SRC, src_mem_1}, {DNNL_ARG_DST, dst_logsoftmax_mem}});
   // engine_stream.wait();
    float tot = 0;
    //for(int i=0;i<((A -> shape[0]) * (A -> shape[1]));i++) cout<<C[i]<<endl;
    for(int i = 0; i < A -> shape[0]; i ++){
        for(int j = 0;j < A -> shape[1]; j ++){
                tot += - C[i * (A -> shape[1]) + j] * ((float*)(B -> data))[i * (B -> shape[1]) + j];
        }
    }
    *((float*)(output -> data)) = tot / (A -> shape[0]);
    delete C;
    return 0;
}
