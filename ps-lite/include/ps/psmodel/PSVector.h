#ifndef PS_PSVECTOR_H_
#define PS_PSVECTOR_H_
#include "ps/ps.h"
#include "ps/psmodel/PSAgent.h"
#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>

using namespace ps;

/*
 * PSVector is a vector on parameter server, supports sparse/dense pull/push 
 * and server-side computation.
 * Operations here are blocked (can be optimized later).
 * PSVector serve as an user interface, the detailed pull/push operation is
 * implemented in @PSAgent.
 */
template <typename Val> class PSVector {
public:
  PSVector(const std::string name, const size_t len) : name(name), length(len) {
    PSAgent<Val>::Get()->DenseVector(name, length);
  }

  /**
   * Derive a new PSVector from this one,
   * and can do PS function on this vector and the new one.  
   * Due to the partition strategy, by default it supports 
   * the derive operation.
   */
  PSVector* derive(const std::string derive_name){
    return new PSVector<Val>(derive_name, length);
  }
  
  void wait(const int timestamp) {
      PSAgent<Val>::Get()->wait(timestamp);
  }

  /**
   * init all zeros
   */
  void initAllZeros(){
      PSAgent<Val>::Get()->initAllZeros(name);
      return;
  }

  /**
   * Elementwise dot over two vectors
   */
  void dot(PSVector* other, Val& res){
      PSAgent<Val>::Get()->vecDot(name, other->name, res);
      return;
  }

  /**
   * Add the value of $other to this PSVector
   */
  void addTo(PSVector* other){
    PSAgent<Val>::Get()->vecAxpy(name, other->name, 1, 0);
    return;
  }

    /**
   * other = this * a + b
   */
  void axpy(PSVector* other, Val a, Val b){
    PSAgent<Val>::Get()->vecAxpy(name, other->name, a, b);
    return;
  }

   /**
   * pull the given indices of this vector,
   * store the results in rets
   */
  void sparsePull(int* offset, Val* rets, const int num_keys, bool inplace=false){
    PSAgent<Val>::Get()->vecSparsePull(name, offset, rets, num_keys, inplace);
    return;
  }

   /**
   * push the given indices of this vector (default is add)
   */
  void sparsePush(int* offset, Val* vals, const int num_keys, bool inplace=false){
    PSAgent<Val>::Get()->vecSparsePush(name, offset, vals, num_keys, inplace);
    return;
  }

  void densePull(Val* rets, const int num_vals, std::vector<int>& timestamp, bool async = false){
    CHECK_EQ(num_vals, length) << "#PSVector size mismatch in densePull";
    if (async == false)
      PSAgent<Val>::Get()->vecDensePull(name, rets, timestamp, async);
   else {
      PSAgent<Val>::Get()->vecDensePull(name, rets, timestamp, async);
   } 
    return;
  }

   /**
   * push the given indices of this vector (default is add)
   */
  void densePush(Val* vals, const int num_vals, std::vector<int>& timestamp, bool async = false){
    CHECK_EQ(num_vals, length) << "#PSVector size mismatch in densePush";
    if (async == false)
      PSAgent<Val>::Get()->vecDensePush(name, vals, timestamp, async);
    else {
      PSAgent<Val>::Get()->vecDensePush(name, vals, timestamp, async);
    }
    return;
  }

private:
  std::string name;
  int length;

};

#endif
