/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef DLSYS_RUNTIME_CPU_DEVICE_API_H_
#define DLSYS_RUNTIME_CPU_DEVICE_API_H_

#include "c_runtime_api.h"
#include "device_api.h"
#include <assert.h>
#include <string>

namespace dlsys {
namespace runtime {

class CPUDeviceAPI : public DeviceAPI {
public:
  void *AllocDataSpace(DLContext ctx, size_t size, size_t alignment, int MEMORY_MANAGE_RULE) final;

  void FreeDataSpace(DLContext ctx, void *ptr, int MEMORY_MANAGE_RULE, size_t memory_size) final;

  void CopyDataFromTo(const void *from, void *to, size_t size,
                      DLContext ctx_from, DLContext ctx_to,
                      DLStreamHandle stream) final;

  void StreamSync(DLContext ctx, DLStreamHandle stream) final;
};

} // namespace runtime
} // namespace dlsys
#endif // DLSYS_RUNTIME_CPU_DEVICE_API_H_
