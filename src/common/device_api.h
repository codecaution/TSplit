/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef DLSYS_RUNTIME_DEVICE_API_H_
#define DLSYS_RUNTIME_DEVICE_API_H_

#include "c_runtime_api.h"
#include <assert.h>
#include <string>

namespace dlsys {
namespace runtime {

class DeviceAPI {
public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {}
  /*!
   * \brief Allocate a data space on device.
   * \param ctx The device context to perform operation.
   * \param size The size of the memory
   * \param alignment The alignment of the memory.
   * \return The allocated device pointer
   */
  virtual void *AllocDataSpace(DLContext ctx, size_t size,
                               size_t alignment, int MEMORY_MANAGE_RULE) = 0;
  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   * \tparam xpu The device mask.
   */
  virtual void FreeDataSpace(DLContext ctx, void *ptr, int MEMORY_MANAGE_RULE, size_t memory_size = 0) = 0;
  /*!
   * \brief copy data from one place to another
   * \param dev The device to perform operation.
   * \param from The source array.
   * \param to The target array.
   * \param size The size of the memory
   * \param ctx_from The source context
   * \param ctx_to The target context
   */
  virtual void CopyDataFromTo(const void *from, void *to, size_t size,
                              DLContext ctx_from, DLContext ctx_to,
                              DLStreamHandle stream) = 0;
  /*!
   * \brief Synchronize the stream
   * \param ctx The context to perform operation.
   * \param stream The stream to be sync.
   */
  virtual void StreamSync(DLContext ctx, DLStreamHandle stream) = 0;
};

} // namespace runtime
} // namespace dlsys
#endif // DLSYS_RUNTIME_DEVICE_API_H_
