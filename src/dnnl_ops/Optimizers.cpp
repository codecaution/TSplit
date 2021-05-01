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
#include <sys/time.h>
#include <omp.h>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"
using namespace dnnl;
using namespace std;

extern "C" int cpu_SGDOptimizerUpdate(const DLArrayHandle param, const DLArrayHandle grad, float learning_rate)
{
    int num = 1;
    for(int i = 0; i < param -> ndim; i++)
        num *= param -> shape[i];

    float* param_data = (float*)(param -> data);
    float* grad_data = (float*)(grad -> data);

    #pragma omp parallel for
    for(int i = 0; i < num; i++)
        param_data[i] -= learning_rate * grad_data[i];
    return 0;
}

extern "C" int cpu_MomentumOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad, DLArrayHandle velocity, float learning_rate, float momentum, bool nesterov)
{
    int num = 1;
    for(int i = 0; i < param -> ndim; i++)
        num *= param -> shape[i];

    float* param_data = (float*)(param -> data);
    float* grad_data = (float*)(grad -> data);
    float* velocity_data = (float*)(velocity -> data);

    if(nesterov){
        #pragma omp parallel for
        for(int i = 0; i < num; i++){
            velocity_data[i] = momentum * (velocity_data[i] - learning_rate * grad_data[i]);
            param_data[i] = param_data[i] + velocity_data[i] - learning_rate * grad_data[i];
        }
    }
    else{
        #pragma omp parallel for
        for(int i = 0;i < num; i++){
            velocity_data[i] = momentum * velocity_data[i] - learning_rate * grad_data[i];
            param_data[i] = param_data[i] + velocity_data[i];
        }
    }
    return 0;
}

extern "C" int cpu_AdaGradOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad, DLArrayHandle acc, float learning_rate, float eps)
{
    int num = 1;
    for(int i = 0; i < param -> ndim; i++)
        num *= param -> shape[i];

    float* param_data = (float*)(param -> data);
    float* grad_data = (float*)(grad -> data);
    float* acc_data = (float*)(acc -> data);

    #pragma omp parallel for
    for(int i = 0; i < num; i++){
        acc_data[i] = acc_data[i] + grad_data[i] * grad_data[i];
        param_data[i] = param_data[i] - learning_rate * grad_data[i] / (sqrtf(acc_data[i]) + eps);
    }
    return 0;
}

extern "C" int cpu_AdamOptimizerUpdate(
    DLArrayHandle param, const DLArrayHandle grad, DLArrayHandle expavg, DLArrayHandle expavgsq,
    float learning_rate, float beta1, float beta2, float beta1t, float beta2t, float eps)
{
    int num = 1;
    for(int i = 0; i < param -> ndim; i++)
        num *= param -> shape[i];

    float* param_data = (float*)(param -> data);
    float* grad_data = (float*)(grad -> data);
    float* expavg_data = (float*)(expavg -> data);
    float* expavgsq_data = (float*)(expavgsq -> data);

    #pragma omp parallel for
    for(int i = 0; i < num; i++){
        expavg_data[i] = beta1 * expavg_data[i] + (1 - beta1) * grad_data[i];
        expavgsq_data[i] = beta2 * expavgsq_data[i] + (1 - beta2) * grad_data[i] * grad_data[i];
        param_data[i] = param_data[i] - learning_rate * (expavg_data[i] / (1 - beta1t)) / (sqrtf(expavgsq_data[i] / (1 - beta2t)) + eps);
    }
    return 0;
}