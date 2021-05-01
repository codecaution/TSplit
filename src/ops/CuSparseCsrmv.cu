#include "gpu_runtime.h"

int CuSparse_DLGpuCsrmv(const DLArrayHandle data_handle,
                   const DLArrayHandle row_handle,
                   const DLArrayHandle col_handle,
                   int nrow, int ncol,
                   bool transpose,
                   const DLArrayHandle input_handle,
                   DLArrayHandle output_handle, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){

  assert (data_handle->ndim == 1);
  /*
  using namespace std;
  std::cout << "Data_hadle_dim:" << data_handle->ndim << std::endl;
  std::cout << "Data_hadle_shape[0]:" << data_handle->shape[0] << std::endl;

  std::cout << "Row_hadle_dim:" << row_handle->ndim << std::endl;
  std::cout << "Row_hadle_shape[0]:" << row_handle->shape[0] << std::endl;

  std::cout << "Col_hadle_dim:" << col_handle->ndim << std::endl;
  std::cout << "Col_hadle_shape[0]:" << col_handle->shape[0] << std::endl;

  std::cout << "nrow:" << nrow << std::endl;
  std::cout << "ncol:" << ncol << std::endl;
  */
  assert (row_handle->ndim == 1);
  assert (col_handle->ndim == 1);
  assert (transpose ? nrow == input_handle->shape[0] : ncol == input_handle->shape[0]);

  int nnz = data_handle->shape[0];
  int dev_id = (data_handle->ctx).device_id;
  cusp_init(dev_id, stream_handle);


  float alpha = 1.0;
  float beta = 0.0;

  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseOperation_t trans = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUSP_CALL(cusparseScsrmv(cusp_map[dev_id], trans,
         nrow, ncol, nnz, (const float*)&alpha, 
         descr, (const float*)data_handle->data, 
         (const int*)row_handle->data, (const int*)col_handle->data, (const float*)input_handle->data,
         (const float*)&beta, (float*)output_handle->data));
  if(p != NULL){
    int size_input1 = 1, size_input2 = 1, size_input3 = 1, size_input4 = 1, size_output = 1;
    for(int i = 0; i < data_handle -> ndim; i++)
        size_input1 *= data_handle -> shape[i];
    for(int i = 0; i < row_handle -> ndim; i++)
        size_input2 *= row_handle -> shape[i];
    for(int i = 0; i < col_handle -> ndim; i++)
        size_input3 *= col_handle -> shape[i];
    for(int i = 0; i < input_handle -> ndim; i++)
        size_input4 *= input_handle -> shape[i];
    for(int i = 0; i < output_handle -> ndim; i++)
        size_output *= output_handle -> shape[i];
    p -> input_memory = 1.0 * (size_input1 + size_input2 + size_input3) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
  }
  return 0;
}