#include "gpu_runtime.h"


void convert_F_to_C_order(float *data, int m, int n)
{
  /*
  convert an F order matrix to C order:
  sample data:
  1, 2, 3, 4
  5, 6, 7, 8
  9, 10, 11, 12
  output:
  1, 4, 7, 10
  2, 5, 8, 11
  3, 6, 9, 12
  */
  size_t workspace_size = m*n;
  float *work_data = (float*)malloc(workspace_size*sizeof(float));
  cudaMemcpy(work_data, data, workspace_size*sizeof(float), cudaMemcpyDeviceToHost);
  float *ans_data = (float*)malloc(workspace_size*sizeof(float));
  for(int k=0; k<workspace_size; k++)
  {
    int i = k%m;
    int j = k/m;
    ans_data[i*n+j] = work_data[k];
  }
  cudaMemcpy(data, ans_data, workspace_size*sizeof(float), cudaMemcpyHostToDevice);
  free(work_data);
  free(ans_data);
  return;
}

int CuSparse_DLGpuCsrmm(const DLArrayHandle data_handle,
                   const DLArrayHandle row_handle,
                   const DLArrayHandle col_handle,
                   int nrow, int ncol,
                   bool transposeA,
                   const DLArrayHandle matB,
                   bool transposeB,
                   DLArrayHandle matC, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  transposeB = ! transposeB;
  //cuSparse limit that A and B cannot transpose at the same time.
  /*
  using namespace std;
  std::cout << "transpose:" << transposeA << transposeB << std::endl;
  */
  assert (!(transposeA == transposeB && transposeA == 1));
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
  
  int workspace_size = matB->shape[0]*matB->shape[1]*sizeof(float);
  float *work_data = (float*)malloc(workspace_size);
  cudaMemcpy(work_data, matB->data, workspace_size, cudaMemcpyDeviceToHost);
  using namespace std;
  for (int i=0; i<workspace_size/sizeof(float); i++)
    std::cout<< "matB[i]=" << work_data[i] << std::endl;
  */

  assert (row_handle->ndim == 1);
  assert (col_handle->ndim == 1);
  assert (matB->ndim == 2);
  assert (matC->ndim == 2);
  int m = nrow;
  int k = ncol;
  int n = matC->shape[1];

  int nnz = data_handle->shape[0];
  int dev_id = (data_handle->ctx).device_id;
  cusp_init(dev_id, stream_handle);

  float alpha = 1.0;
  float beta = 0.0;

  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseOperation_t transA = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  int ldb = matB->shape[1];
  int ldc = matC->shape[0];
  CUSP_CALL(cusparseScsrmm2(cusp_map[dev_id], transA, transB, m, n, k, nnz,
                (const float*)&alpha, descr, (const float*)data_handle->data, 
                (const int*)row_handle->data, (const int*)col_handle->data,
                (const float*)matB->data, (int)ldb, (const float*)&beta,
                (float*)matC->data, (int)ldc));
  convert_F_to_C_order((float*)matC->data, matC->shape[0], matC->shape[1]);
  if(p != NULL){
    int size_input1 = 1, size_input2 = 1, size_input3 = 1, size_output = 1;
    for(int i = 0; i < data_handle -> ndim; i++)
        size_input1 *= data_handle -> shape[i];
    for(int i = 0; i < row_handle -> ndim; i++)
        size_input2 *= row_handle -> shape[i];
    for(int i = 0; i < col_handle -> ndim; i++)
        size_input3 *= col_handle -> shape[i];
    for(int i = 0; i < matC -> ndim; i++)
        size_output *= matC -> shape[i];
    p -> input_memory = 1.0 * (size_input1 + size_input2 + size_input3) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
  }
  return 0;
}