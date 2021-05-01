#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <omp.h>
#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"

extern "C" int cpu_Transpose(const DLArrayHandle in_arr, DLArrayHandle out_arr) {
	float* input = (float*)(in_arr->data);
	float* output = (float*)(out_arr->data);
	int row = in_arr->shape[0];
	int col = in_arr->shape[1];
	int ind = 0;

    #pragma omp parallel for
	for (int i = 0; i < col * row; i++)
		output[i] = input[ (i/row) + col * (i%row) ];
	return 0;
}