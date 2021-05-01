#ifndef PS_PSF_HANDLE_H_
#define PS_PSF_HANDLE_H_

#include "ps/base.h"
#include "ps/kvmeta.h"
#include "ps/kvpairs.h"
#include "ps/kvserver.h"
#include "ps/kvworker.h"
#include "ps/psf/PSFunc.h"
#include "ps/simple_app.h"
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
namespace ps {
/**
 * \brief an example handle adding pushed kv into store
 */
template <typename Val> struct KVServerDefaultHandle {
  void operator()(const KVMeta &req_meta, const KVPairs<Val> &req_data,
                  KVServer<Val> *server) {
    size_t n = req_data.keys.size();

    KVPairs<Val> res;
    if (!req_meta.pull) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n, 0);
    }
    for (size_t i = 0; i < n; ++i) {
      Key key = req_data.keys[i];
      if (req_meta.push) {
        store[key] += req_data.vals[i];
      }
      if (req_meta.pull) {
        res.vals[i] = store[key];
      }
    }
    server->Response(req_meta, res);
  }
  std::unordered_map<Key, Val> store;
};

/**
 * \brief used in ML part for sparse/dense pull, push.
 *        keys is used for the key of one partition.
 *        lens is used as the offset of the keys.
 *        vals is vals.
 *        One key (two keys for binary op) per request in Athena.
 *        Is it ok in a lock-free manner? By @Zhipeng
 */
template <typename Val> struct KVServerMatrixHandle {
  void operator()(const KVMeta &req_meta, const KVPairs<Val> &req_data,
                  KVServer<Val> *server) {
    KVPairs<Val> res;
    PsfType psftype = req_meta.psftype;
    // TODO: do not return the keys and the lens since they are already stored
    // on workers.
    res.keys = req_data.keys;
    res.lens = req_data.lens;

    switch (psftype) {
    case PsfType::DensePull: {
      // one key per request.
      // with response result
      Key k = req_data.keys[0];
      size_t len = req_data.lens[0];
      if (store.find(k) != store.end()) {
        auto &value_set = store[k];
        CHECK_EQ(len, value_set.size()) << " size mismatch in DensePull";
        res.vals.resize(value_set.size());
        for (size_t j = 0; j < value_set.size(); j++)
          res.vals[j] = value_set[j];
      } else {
        LG << "Key does not exist on PS in DensePull" << k;
      }
      break;
    }
    case PsfType::DensePush: {
      // one key per request.
      // no response result
      Key k = req_data.keys[0];
      size_t len = req_data.lens[0];
      if (store.find(k) != store.end()) {
        CHECK_EQ(len, store[k].size()) << k << " " << len <<" " << store[k].size() <<" size mismatch in DensePush";
      } else {
        store[k].assign(len, 0);
        LG << "Key does not exist on PS in DensePush: " << k;
      }
      auto &value_set = store[k];
      for (size_t j = 0; j < value_set.size(); j++)
        value_set[j] += req_data.vals[j];
      break;
    }
    case PsfType::PushPull: {
      // one key per request.
      // with response result
      Key k = req_data.keys[0];
      size_t len = req_data.lens[0];
      if (store.find(k) != store.end()) {
        CHECK_EQ(len, store[k].size()) << " size mismatch in DensePush";
      } else {
        store[k].assign(len, 0);
        LG << "Key does not exist on PS in DensePush: " << k;
      }
      res.vals.resize(len);
      auto &value_set = store[k];
      auto &resVal = res.vals;
      for (size_t j = 0; j < value_set.size(); j++) {
        value_set[j] += req_data.vals[j];
        resVal[j] = value_set[j];
      }
      break;
    }
    case PsfType::InitAllZeros: {
      // one key per request.
      // no response result
      Key k = req_data.keys[0];
      size_t len = req_data.lens[0];
      if (store.find(k) != store.end()) {
        CHECK_EQ(len, store[k].size()) << " size mismatch in DensePush";
        LG << "init PSVector with key: " << k << ", key already existed";
      } else {
        store[k].assign(len, 0);
        LG << "init PSVector with key: " << k << " , length: " << len;
      }
      break;
    }
    case PsfType::SparsePush: {
      // we use length as the offset, i.e., #length = #vals.
      // no response result
      Key k = req_data.keys[0];
      auto &offset = req_data.lens;
      auto &vals = req_data.vals;
      CHECK_EQ(vals.size(), offset.size())
          << " in Psf::SparsePush check failed,"
          << " size of vals is " << vals.size() << " size of lens is "
          << offset.size();
      if (store.find(k) != store.end()) {
        auto &value_set = store[k];
        for (size_t j = 0; j < offset.size(); j++) {
          value_set[offset[j]] += vals[j];
        }
      } else {
        // error, the key does not exist on PS.
        LF << "[Error] The pushed key: " << k
           << " does not exist on PS in SparsePush.";
      }
      break;
    }
    case PsfType::SparsePull: {
      // we use length as the offset, i.e., #length = #vals.
      // with response result
      Key k = req_data.keys[0];
      auto &offset = req_data.lens;
      if (store.find(k) != store.end()) {
        res.vals.resize(offset.size());
        auto &value_set = store[k];
        for (size_t j = 0; j < offset.size(); j++) {
          res.vals[j] = value_set[offset[j]];
        }
      } else {
        // error, the key does not exist on PS.
        LF << "[Error] The pulled key: " << k
           << " does not exist on PS in SparsePull.";
      }
      break;
    }
    case PsfType::Dot: {
      // in this dot function, there are two keys in each request
      // with response result, a scalar.
      auto keys = req_data.keys;
      CHECK_EQ(keys.size(), 2)
          << " the size of keys does not equal to 2 in Psf::Dot";
      res.vals.resize(1);
      auto key1 = keys[0];
      auto key2 = keys[1];
      if (store.find(key1) != store.end() && store.find(key2) != store.end()) {
        auto &value_set1 = store[key1];
        auto &value_set2 = store[key2];
        Val sum = 0;
        for (size_t j = 0; j < value_set1.size(); j++) {
          sum += value_set1[j] * value_set2[j];
        }
        res.vals[0] = sum;
      } else {
        // error, the key does not exist on PS.
        LF << "[Error] The pulled key: " << key1 << " and " << key2
           << " does not exist on PS-dot.";
      }
      break;
    }
    case PsfType::Axpy: {
      // in this axpy function (y += ax + b), there are two keys in each request
      // and b and stored in vals
      // no response result.
      auto &keys = req_data.keys;
      auto &vals = req_data.vals;
      CHECK_EQ(keys.size(), 2)
          << " the size of keys does not equal to 2 in Psf::Axpy";
      CHECK_EQ(vals.size(), 2) << " the size of vals [should be `a` and `b`] "
                                  "does not equal to 2 in Psf::Axpy";
      auto key1 = keys[0];
      auto key2 = keys[1];
      auto a = vals[0];
      auto b = vals[1];
      if (store.find(key1) != store.end() && store.find(key2) != store.end()) {
        auto &value_set1 = store[key1];
        auto &value_set2 = store[key2];
        for (size_t j = 0; j < value_set1.size(); j++) {
          value_set2[j] += a * value_set1[j] + b;
        }
      } else {
        // error, the key does not exist on PS.
        LF << "[Error] The keys: " << key1 << " and " << key2
           << " does not exist on PS-Axpy.";
      }
      break;
    }
    case PsfType::PushCols: {
      // request by a PSMatrix, push multiple columns to PS.
      // we use length as the offset, num_rows is appended at the tail of
      // offsets. no response result
      Key k = req_data.keys[0];
      auto &offset = req_data.lens;
      auto &vals = req_data.vals;
      int num_rows = offset.back();
      int num_elements = num_rows * (offset.size() - 1);
      if (store.find(k) != store.end()) {
        auto &value_set = store[k];
        int cnt = 0;
        for (size_t j = 0; j < offset.size() - 1; j++) {
          for (int p = 0; p < num_rows; p++) {
            value_set[offset[j] * num_rows + p] += vals[cnt];
            cnt++;
          }
        }
        CHECK_EQ(cnt, num_elements) << "size does not match in Psf:PushCols";
      } else {
        // error, the key does not exist on PS.
        LF << "[Error] The pushed key: " << k
           << " does not exist on PS in PushCols.";
      }
      break;
    }
    case PsfType::PullCols: {
      // request by a PSMatrix, pull multiple columns.
      // we use length as the offset, i.e., #length = #vals.
      // with response result
      Key k = req_data.keys[0];
      auto &offset = req_data.lens;
      int num_rows = offset.back();
      int res_size = num_rows * (offset.size() - 1);
      if (store.find(k) != store.end()) {
        res.vals.resize(res_size);
        auto &value_set = store[k];
        int cnt = 0;
        for (size_t j = 0; j < offset.size() - 1; j++) {
          for (int p = 0; p < num_rows; p++) {
            res.vals[cnt] = value_set[offset[j] * num_rows + p];
            cnt++;
          }
        }
        CHECK_EQ(cnt, res_size) << "size does not match in Psf:PullCols";
      } else {
        // error, the key does not exist on PS.
        LF << "[Error] The pulled key: " << k
           << " does not exist on PS in PullCols.";
      }
      break;
    }
    default: {
      break;
    }
    }
    server->Response(req_meta, res);
  }
  std::unordered_map<Key, std::vector<Val>> store;
};
} // namespace ps

#endif
