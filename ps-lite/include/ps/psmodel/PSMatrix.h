#ifndef PS_PSMATRIX_H_
#define PS_PSMATRIX_H_
#include "ps/ps.h"
#include "ps/psmodel/PSAgent.h"
#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>

using namespace ps;

/*
 * PSMatrix is a matrix on parameter server partitioned by cols and
 * supports pullRows/pushRows.
 * Operations here are blocked (can be optimized later).
 * PSMatrix serve as an user interface, the detailed pull/push operation is
 * implemented in @PSAgent.
 */
template <typename Val> class PSMatrix {
public:
  PSMatrix(const std::string name, const size_t rows, const size_t cols)
      : name(name), rows(rows), cols(cols) {
    PSAgent<Val>::Get()->DenseMatrix(name, rows, cols);
  }

  /**
   * init all zeros
   */
  void initAllZeros() {
    PSAgent<Val>::Get()->initAllZeros(name, rows);
    return;
  }

  /**
   * \brief pull rows of the PSMatrix from PS.
   * @param offset the colId of the PSMatrix, the colId should be ordered in an increasing order.
   * @param rets store the pulled results in rets.
   * @param num_keys number of columns to pull from PS.
   */
  void pullCols(int *offset, Val *rets, const int num_keys, bool inplace=false) {
    PSAgent<Val>::Get()->matPullCols(name, offset, rets, num_keys, rows, inplace);
    return;
  }

  /**
   * \brief push rows of the PSMatrix to PS.
   * @param offset the colId of the PSMatrix, the colId should be ordered in an increasing order.
   * @param vals the vals to push to PS.
   * @param num_keys number of columns to pull from PS.
   */
  void pushCols(int *offset, Val *vals, const int num_keys, bool inplace=false) {
    PSAgent<Val>::Get()->matPushCols(name, offset, vals, num_keys, rows, inplace);
    return;
  }

private:
  std::string name;
  int rows;
  int cols;
};

#endif
