#ifndef DLSYS_P_
#define DLSYS_P_
#include<iostream>
#ifdef __cplusplus
#define DLSYS_EXTERN_C extern "C"
#else
#define DLSYS_EXTERN_C
#endif

#include <stddef.h>
#include <stdint.h>
DLSYS_EXTERN_C{
    struct Profiler{
        float time;
        float input_memory;
        float output_memory;
        float workspace_memory;

        Profiler(): time(-1), input_memory(-1),output_memory(-1),workspace_memory(-1) {
//            std::cout << time << input_memory << output_memory << workspace_memory << std::endl;
        }
    };
}
#endif