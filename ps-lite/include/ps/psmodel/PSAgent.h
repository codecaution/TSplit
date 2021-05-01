#ifndef PS_PSAGENT_H_
#define PS_PSAGENT_H_
#include "ps/ps.h"
#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>

using namespace ps;

/*
 * A singleton object for pulling or push to PS.
 * Since we enable sparse pull/push in PSVector and the length of each val is
 * one, thus the $lens in @kvpairs is not useful. As a result, we use $lens to
 * store the offset of each vector. for example, key=1000, lens = {1,2,3}, then
 * we are accessing elements with ids as {1000+1, 1000+2, 1000+3}
 */
template <typename Val> class PSAgent {
private:
  /* The KVWorker used to pull/push. */
  KVWorker<Val> _kvworker;
  /* split a tensor into multiple pieces. [node_name --> splitted_dl_array_keys] */
  std::unordered_map<std::string, std::vector<Key>> _id2key;
  /* [node_name --> splitted_dl_array_lens] */
  std::unordered_map<std::string, std::vector<int>> _id2length;
  /* used to generate the key of the tensors.
   * The map is: node.id --> globalId + i-th PS.start(). */
  Key _globalId = 0;
  /* the max number of floats in a single push, a hyperparameter in
   * partitionPull. */
  std::shared_ptr<ThreadPool> _thread_pool;
  /* concurrency for push/pull message */
  int _thread_num = 5;

  PSAgent() : _kvworker(0, 0) {
    _thread_pool = std::shared_ptr<ThreadPool>(new ThreadPool(_thread_num));
  }

public:
  static PSAgent *Get() {
    static PSAgent e;
    return &e;
  }

  void barrier(){
    Postoffice::Get()->Barrier(0, kWorkerGroup);
  }

  void wait(int timestamp) {
      _kvworker.Wait(timestamp);
  }
  /**
   * \brief init the meta information about this tensor on PS.
   *        the meta data is stored on each worker.
   * \param name the name of the input tensor
   * \param cols the #columns of the input tensor, the tensors are partitioned by cols.
   */
  void insertTensorMeta(const std::string name, const int cols) {
    if (_id2key.find(name) != _id2key.end() &&
        _id2length.find(name) != _id2length.end())
      return;
    // we insert the meta data of this tensor
    const std::vector<Range> &server_range =
        Postoffice::Get()->GetServerKeyRanges();
    int server_num = (int)server_range.size();

    // partition the tensor to parameter servers
    std::vector<Key>& keys = _id2key[name];
    std::vector<int>& lens = _id2length[name];
    int len_per_server = cols / server_num;
    int len_last_server = cols - len_per_server * (server_num - 1);
    for (int server_id = 0; server_id < server_num - 1; server_id ++) {
      keys.push_back(_globalId + server_range[server_id].begin());
      lens.push_back(len_per_server);
    }
    // last server
    keys.push_back(_globalId + server_range[server_num - 1].begin());
    lens.push_back(len_last_server);
    _globalId++;
  }

  /**
   * A vector is partitioned by cols.
   */
  void DenseVector(const std::string name, const int length) {
    insertTensorMeta(name, length);
  }

  void initAllZeros(const std::string name, const int rows = 1) {
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::future<int>> ts(keys.size());

    /* send init request to each partition */
    for (int i = 0; i < (int)keys.size(); i++) {
      ts[i] = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, const int tmp_len) -> int {
            /* no priority here. */
            return (_kvworker.ZPush(SArray<Key>({tmp_tensor_key}), SArray<Val>({}),
                                    SArray<int>({tmp_len}), 0, nullptr, 0,
                                    PsfType::InitAllZeros));
          },
          keys[i], lens[i] * rows);
    }
    // do block PS operation
    for (auto &t : ts)
      _kvworker.Wait(t.get());
    return;
  }

  /**
   * \brief PSVector: push <Key, Val> pairs to PS. Here the offset is the offset
   *        of keys comparing with zero.
   * \param name name of the PSVector
   * \param offsets the keys of the PSVector. Stored in @kvpairs.lens
   * \param vals the vals of pushed vals
   * \param isInpalce whether we do zero-copy on offsets.
   */
  void vecSparsePush(const std::string name, int* offsets,
            Val* vals, const int num_offsets, bool inplace) {
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::pair<bool, std::future<int>>> ts(keys.size());
    int start = 0, end = 0;
    int cur_len = 0;
    int* cp_offsets;
    if(inplace){
      cp_offsets = offsets;
    }else{
      cp_offsets = new int[num_offsets];
      memcpy(cp_offsets, offsets, num_offsets * sizeof(int));
    }
    std::vector<SArray<int>> _piece_offset(keys.size());
    std::vector<SArray<Val>> _piece_val(keys.size());
    /* send push request to each partition according to the offsets. */
    for (int i = 0; i < (int)keys.size(); i++) {
      while (end < num_offsets && offsets[end] < cur_len + lens[i]) {
        cp_offsets[end] -= cur_len;
        end++;
      }
      if(start == end){
        // no need to send a request for this partition.
        cur_len += lens[i];
        start = end;
        ts[i].first = false;
        continue;
      }
      _piece_offset[i] = SArray<int>(cp_offsets + start, end - start);
      _piece_val[i] = SArray<Val>(vals + start, end - start);
      ts[i].first = true;
      ts[i].second = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, const SArray<Val> tmp_vals,
              const SArray<int> tmp_offsets) -> int {
            /* no priority here. */
            return (_kvworker.ZPush(SArray<Key>({tmp_tensor_key}), tmp_vals,
                                    tmp_offsets, 0, nullptr, 0,
                                    PsfType::SparsePush));
          },
          keys[i], _piece_val[i], _piece_offset[i]);
      start = end;
      cur_len += lens[i];
    }
    // do block PS operation
    for (auto &t : ts){
      if(t.first)
        _kvworker.Wait(t.second.get());
    }
    if(!inplace){
      delete []cp_offsets;
    }
    return;
  }

  /**
   * \brief PSVector: pull <Key, Val> pairs from PS. Here the offset is the offset
   *        of keys comparing with zero.
   * \param name name of the PSVector
   * \param offsets the keys of the PSVector. Stored in @kvpairs.lens
   * \param vals the vals of pulled vals
   * \param inplace whether we do inplace operation on the offsets.
   */
  void vecSparsePull(const std::string name, int* offsets,
            Val* vals, const int num_offsets, bool inplace) {
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::pair<bool, std::future<int>>> ts(keys.size());
    int start = 0, end = 0;
    int cur_len = 0;

    int* cp_offsets;
    if(inplace){
      cp_offsets = offsets;
    }else{
      cp_offsets = new int[num_offsets];
      memcpy(cp_offsets, offsets, num_offsets * sizeof(int));
    }
    std::vector<SArray<int>> _piece_offset(keys.size());
    std::vector<SArray<Val>> _piece_val(keys.size());
    /* send push request to each partition */
    for (int i = 0; i < (int)keys.size(); i ++) {
      while (end < num_offsets && offsets[end] < cur_len + lens[i]) {
        cp_offsets[end] -= cur_len;
        end++;
      }
      if(start == end){
        // no need to send a request for this partition.
        cur_len += lens[i];
        start = end;
        ts[i].first = false;
        continue;
      }
      // zero copy from Vector and zero copy slice.
      _piece_offset[i] = SArray<int>(cp_offsets + start, end - start);
      _piece_val[i] = SArray<Val>(vals + start, end - start);
      ts[i].first = true;
      ts[i].second = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, SArray<Val>& tmp_vals, 
                SArray<int>& tmp_offsets) -> int {
            /* no priority here. */
            return (_kvworker.ZPull(SArray<Key>({tmp_tensor_key}), &tmp_vals,
                                    &tmp_offsets, 0, nullptr, 0,
                                    PsfType::SparsePull));
          },
          keys[i], _piece_val[i], _piece_offset[i]);
      start = end;
      cur_len += lens[i];
    }
    // do block PS operation
    for (auto &t : ts){
      if(t.first)
        _kvworker.Wait(t.second.get());
    }
    if(!inplace){
      delete []cp_offsets;
    }
    return;
  }

  /**
   * \brief PSVector: pull <Key, Vals> pairs from PS.
   * \param name name of the PSVector
   * \param vals the vals of pullsh vals
   */
  void vecDensePush(const std::string name, Val* vals, std::vector<int>& timestamp, bool async = false){
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::future<int>> ts(keys.size());
    std::vector<SArray<Val>> _piece_val(keys.size());
    /* send push request to each partition according to the offsets. */
    int cur_len = 0;
    for (int i = 0; i < (int)keys.size(); i++) {
      int _key = keys[i];
      int _len = lens[i];
      _piece_val[i] = SArray<Val>(vals + cur_len, _len);
      cur_len += _len;
      ts[i] = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, const SArray<Val> tmp_vals,
              const SArray<int> tmp_tensor_len) -> int {
            /* no priority here. */
            return (_kvworker.ZPush(SArray<Key>({tmp_tensor_key}), tmp_vals,
                                    tmp_tensor_len, 0, nullptr, 0,
                                    PsfType::DensePush));
          },
          _key, _piece_val[i], SArray<int>({_len}));
    }
    if (async == false){
    // do block PS operation
      for (auto &t : ts){
        _kvworker.Wait(t.get());
      }
    } else {
        timestamp.reserve(ts.size());
        for (auto & t : ts)
          timestamp.push_back(t.get());  
    }
    return;
  }

  void vecDensePull(const std::string name, Val* vals, std::vector<int>& timestamp, bool async = false){
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::future<int>> ts(keys.size());
    std::vector<SArray<Val>> _piece_val(keys.size());
    int cur_len = 0; 
    /* send pull request to each partition */
    for (int i = 0; i < (int)keys.size(); i ++) {
      int _key = keys[i];
      int _len = lens[i];
      _piece_val[i] = SArray<Val>(vals + cur_len, _len);
      SArray<int> slens({_len});
      cur_len += _len;
      ts[i] = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, SArray<Val>& tmp_vals, 
                SArray<int>& tmp_tensor_len) -> int {
            /* no priority here. */
            return (_kvworker.ZPull(SArray<Key>({tmp_tensor_key}), &tmp_vals,
                                    &tmp_tensor_len, 0, nullptr, 0,
                                    PsfType::DensePull));
          },
          _key, _piece_val[i], slens);
    }
    if (async == false){
    // do block PS operation
      for (auto &t : ts){
        _kvworker.Wait(t.get());
      }
    } else {
      timestamp.reserve(ts.size());
      for (auto &t : ts){
        timestamp.push_back(t.get());
      }
    } 
    return;
  }

  /**
   * \brief Elementwise dot over two vectors
   * \param name1 the name of two PSVectors.
   * \param name2 
   * \param res stores the dot of these two vectors
   */
  void vecDot(const std::string name1, const std::string name2, Val& res) { 
    const std::vector<Key> &keys1 = _id2key[name1];
    const std::vector<int> &lens1 = _id2length[name1];
    const std::vector<Key> &keys2 = _id2key[name2];
    const std::vector<int> &lens2 = _id2length[name2];
    CHECK_EQ(keys1.size(), keys2.size()) << " shape mismatch in PSVector Dot"; 
    Val resArr[keys1.size()];
    std::vector<std::future<int>> ts(keys1.size());
    std::vector<SArray<Val>> _piece_val(keys1.size());
    /* send push request to each partition */
    // no input for each tensor, just keys, no lens, no vals.
    for (int i = 0; i < (int)keys1.size(); i ++) {
      auto k1 = keys1[i];
      auto k2 = keys2[i];
      SArray<Key> skeys({k1, k2});
      SArray<int> slens({});
      _piece_val[i] = SArray<Val>(resArr + i, 1);
      CHECK_EQ(lens1[i], lens2[i]) << " shape mismatch in PSVector Dot"; 
      ts[i]= _thread_pool->Enqueue(
          [this](const SArray<Key> _skeys, SArray<Val> _svals, 
            SArray<int> _slens) -> int {
            /* no priority here. */
            return (_kvworker.ZPull(_skeys, &_svals, &_slens, 0, nullptr, 0,
                                    PsfType::Dot));
          },
          skeys, _piece_val[i], slens);
    }
    // do block PS operation
    for (auto &t : ts){
        _kvworker.Wait(t.get());
    }
    res = 0;
    for(int i = 0; i < (int)keys1.size(); i ++){
      res += resArr[i];
    }
    return;
  }

  /**
   * \brief Add the value of $other to this PSVector, name2 = name1 * a + b
   */
  void vecAxpy(const std::string name1, const std::string name2, Val a, Val b){
    // to implement.
    const std::vector<Key> &keys1 = _id2key[name1];
    const std::vector<int> &lens1 = _id2length[name1];
    const std::vector<Key> &keys2 = _id2key[name2];
    const std::vector<int> &lens2 = _id2length[name2];
    CHECK_EQ(keys1.size(), keys2.size()) << " shape mismatch in PSVector Axpy"; 
    std::vector<std::future<int>> ts(keys1.size());
    /* send push request to each partition */
    // no input for each tensor, just keys, no lens, no vals.
    for (int i = 0; i < (int)keys1.size(); i ++) {
      auto k1 = keys1[i];
      auto k2 = keys2[i];
      SArray<Key> skeys({k1, k2});
      SArray<int> slens({});
      SArray<Val> svals({a, b});
      CHECK_EQ(lens1[i], lens2[i]) << " shape mismatch in PSVector Dot"; 
      ts[i]= _thread_pool->Enqueue(
          [this](const SArray<Key> _skeys, SArray<Val> _svals, 
            SArray<int> _slens) -> int {
            /* no priority here. */
            return (_kvworker.ZPush(_skeys, _svals, _slens, 0, nullptr, 0,
                                    PsfType::Axpy));
          },
          skeys, svals, slens);
    }
    // do block PS operation
    for (auto &t : ts){
        _kvworker.Wait(t.get());
    }
    return;
  }

    /**
   * A Matrix is partitioned by rows.
   */
  void DenseMatrix(const std::string name, const int rows, const int cols){
    insertTensorMeta(name, cols);
  }

  /**
   * \brief push multiple columns to PS. Here the offset is the offset
   *        of keys comparing with zero.
   * \param name name of the PSVector
   * \param offsets the keys of the PSVector
   * \param vals the vals of pushed vals
   * \param rows the number of rows
   * \param inplace whether we do inplace operation on offsets
   */
  void matPushCols(const std::string name, int* offsets,
            Val* vals, const int num_offsets, const int num_rows, bool inplace) {
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::pair<bool, std::future<int>>> ts(keys.size());
    int start = 0, end = 0;
    int cur_len = 0;

    int* cp_offsets;
    if(inplace){
      cp_offsets = offsets;
    }else{
      cp_offsets = new int[num_offsets];
      memcpy(cp_offsets, offsets, num_offsets * sizeof(int));
    }
    std::vector<SArray<int>> _piece_offset(keys.size());
    std::vector<SArray<Val>> _piece_val(keys.size());
    /* send push request to each partition according to the offsets. */
    for (int i = 0; i < (int)keys.size(); i++) {
      while (end < num_offsets && offsets[end] < cur_len + lens[i]) {
        cp_offsets[end] -= cur_len;
        end++;
      }
      if(start == end){
        // no need to send a request for this partition.
        cur_len += lens[i];
        start = end;
        ts[i].first = false;
        continue;
      }
      _piece_offset[i] = SArray<int>(cp_offsets + start, end - start);
      // add number of rows info at the last of pieceoffset.
      _piece_offset[i].push_back(num_rows);
      _piece_val[i] = SArray<Val>(vals + start * num_rows, (end - start) * num_rows);
      ts[i].first = true;
      ts[i].second = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, const SArray<Val> tmp_vals,
              const SArray<int> tmp_offsets) -> int {
            /* no priority here. */
            return (_kvworker.ZPush(SArray<Key>({tmp_tensor_key}), tmp_vals,
                                    tmp_offsets, 0, nullptr, 0,
                                    PsfType::PushCols));
          },
          keys[i], _piece_val[i], _piece_offset[i]);
      start = end;
      cur_len += lens[i];
    }
    // do block PS operation
    for (auto &t : ts){
      if(t.first)
        _kvworker.Wait(t.second.get());
    }
    if(!inplace){
      delete []cp_offsets;
    }
    return;
  }

  /**
   * \brief pull multiple columns from the PS. Here the offset is the offset
   *        of keys comparing with zero.
   * \param name name of the PSVector
   * \param offsets the keys of the PSVector
   * \param vals the vals of pushed vals
   * \param rows the number of rows
   * \param inplace whether we do inplace operation on offsets
   */
  void matPullCols(const std::string name, int* offsets,
            Val* vals, const int num_offsets, const int num_rows, bool inplace) {
    const std::vector<Key> &keys = _id2key[name];
    const std::vector<int> &lens = _id2length[name];
    std::vector<std::pair<bool, std::future<int>>> ts(keys.size());
    int start = 0, end = 0;
    int cur_len = 0;

    int* cp_offsets;
    if(inplace){
      cp_offsets = offsets;
    }else{
      cp_offsets = new int[num_offsets];
      memcpy(cp_offsets, offsets, num_offsets * sizeof(int));
    }
    std::vector<SArray<int>> _piece_offset(keys.size());
    std::vector<SArray<Val>> _piece_val(keys.size());
    /* send push request to each partition */
    for (int i = 0; i < (int)keys.size(); i ++) {
      while (end < num_offsets && offsets[end] < cur_len + lens[i]) {
        cp_offsets[end] -= cur_len;
        end++;
      }
      if(start == end){
        // no need to send a request for this partition.
        cur_len += lens[i];
        start = end;
        ts[i].first = false;
        continue;
      }
      _piece_offset[i] = SArray<int>(cp_offsets + start, end - start);
      // add number of rows info at the last of pieceoffset.
      _piece_offset[i].push_back(num_rows);
      _piece_val[i] = SArray<Val>(vals + start * num_rows, (end - start) * num_rows);
      ts[i].first = true;
      ts[i].second = _thread_pool->Enqueue(
          [this](const Key tmp_tensor_key, SArray<Val>& tmp_vals, 
                SArray<int>& tmp_offsets) -> int {
            /* no priority here. */
            return (_kvworker.ZPull(SArray<Key>({tmp_tensor_key}), &tmp_vals,
                                    &tmp_offsets, 0, nullptr, 0,
                                    PsfType::PullCols));
          },
          keys[i], _piece_val[i], _piece_offset[i]);
      start = end;
      cur_len += lens[i];
    }
    // do block PS operation
    for (auto &t : ts){
      if(t.first)
        _kvworker.Wait(t.second.get());
    }
    if(!inplace){
      delete []cp_offsets;
    }
    return;
  }

};
#endif
