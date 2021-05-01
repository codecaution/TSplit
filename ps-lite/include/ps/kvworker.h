#ifndef PS_KV_WORKER_H_
#define PS_KV_WORKER_H_
#include "ps/base.h"
#include "ps/kvmeta.h"
#include "ps/kvpairs.h"
#include "ps/psf/PSFunc.h"
#include "ps/simple_app.h"
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
namespace ps {

/**
 * \brief A worker node that can \ref Push (\ref Pull) key-value pairs to (from)
 * server nodes
 *
 * \tparam Val the type of value, which should be primitive types such as
 * int32_t and float
 */
template <typename Val> class KVWorker : public SimpleApp {
public:
  /** avoid too many this-> */
  using SimpleApp::obj_;
  /**
   * \brief callback function for \ref Push and \ref Pull
   *
   * It is called by the data receiving thread of this instance when the push or
   * pull is actually finished. Namely the kv pairs have already written into
   * servers' data structure or the kv pairs have already pulled back.
   */
  using Callback = std::function<void()>;

  /**
   * \brief constructor
   *
   * \param app_id the app id, should match with \ref KVServer's id
   * \param customer_id the customer id which is unique locally
   */
  explicit KVWorker(int app_id, int customer_id) : SimpleApp() {
    using namespace std::placeholders;
    obj_ = new Customer(app_id, customer_id,
                        std::bind(&KVWorker<Val>::Process, this, _1));
  }

  /** \brief deconstructor */
  virtual ~KVWorker() {
    delete obj_;
    obj_ = nullptr;
  }

  /**
   * \brief Pushes a list of key-value pairs to all server nodes.
   *
   * This function pushes a KV list specified by \a keys and \a vals to all
   * server nodes.
   *
   * Sample usage: the following codes push two KV pairs `{1, (1.1, 1.2)}` and
   * `{3, (3.1,3.2)}` to server nodes, where the value is a length-2 float
   * vector \code KVWorker<float> w; std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals = {1.1, 1.2, 3.1, 3.2};
   *   w.Push(keys, vals);
   * \endcode
   *
   * If \a lens is given, then the value can be various length. See
   * \ref KVPairs for more information.
   *
   * The KV list is partitioned and sent based on the key range each server
   * maintaining. This function returns without waiting the data are sent
   * actually. Instead, use either \ref Wait or the callback to know when
   * finished. This function is thread-safe.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the according values
   * @param lens optional, lens[i] stores the value length of the \a
   * i-th KV pair
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the push is finished.
   * @return the timestamp of this request
   */
  int Push(const std::vector<Key> &keys, const std::vector<Val> &vals,
           const std::vector<int> &lens, int cmd = 0,
           const Callback &cb = nullptr, int priority = 0,
           PsfType psftype = PsfType::DensePush) {
    return ZPush(SArray<Key>(keys), SArray<Val>(vals), SArray<int>(lens), cmd,
                 cb, priority, psftype);
  }

  /**
   * \brief Pulls the values associated with the keys from the server nodes
   *
   * This function pulls the values of the keys specified in \a keys from the
   * server nodes. The format is same to \ref KVPairs
   *
   * Sample usage: the following codes pull the values of keys \a 1 and \a 3
   * from the server nodes.
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals;
   *   w.Pull(keys, &vals);
   * \endcode
   *
   * It's a non-blocking call. The actual pulling is finished,
   * namely \a vals (and \a lens) is filled with pulled values, only
   * if \ref Wait returns or the callback is called.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the buffer for the pulled values. It can be 0 size.
   * @param lens optional buffer for the value length. If set, it can be 0 size.
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the pull is finished.
   * @return the timestamp of this request
   */
  int Pull(const std::vector<Key> &keys, std::vector<Val> *vals,
           std::vector<int> *lens, int cmd = 0,
           const Callback &cb = nullptr, int priority = 0,
           PsfType psftype = PsfType::DensePull) {
    SArray<Key> skeys(keys);
    int ts = AddPullCB(skeys, vals, lens, cmd, cb, psftype);
    KVPairs<Val> kvs;
    kvs.keys = skeys;
    kvs.priority = priority;
    Send(ts, false, true, cmd, psftype, kvs);
    return ts;
  }

  /**
   * \brief Pushes and Pulls a list of key-value pairs to and from the server
   * nodes.
   *
   * This function pushes the values of the keys specified in \a keys to the
   * server nodes and subsequently pulls and updates the values in \a vals.
   *
   * Sample usage: the following code pushes and pulls the values of keys
   * \a 1 and \a 3 to and from the server nodes.
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals;
   *   w.PushPull(keys, &vals);
   * \endcode
   *
   * It's a non-blocking call. The actual pulling is finished,
   * namely \a vals (and \a lens) is filled with pulled values, only
   * if \ref Wait returns or the callback is called.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the according values
   * @param outs the buffer for the pulled values. It can be 0 size.
   * @param lens optional buffer for the value length. If set, it can be 0 size.
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the pull is finished.
   * @return the timestamp of this request
   */
  int PushPull(const std::vector<Key> &keys, const std::vector<Val> &vals,
               std::vector<Val> *outs, std::vector<int> *lens,
               int cmd = 0, const Callback &cb = nullptr, int priority = 0,
               PsfType psftype = PsfType::PushPull) {
    CHECK_NOTNULL(outs);
    if (outs->empty())
      outs->resize(vals.size());
    else
      CHECK_EQ(vals.size(), outs->size());

    SArray<Key> skeys(keys);
    SArray<Val> svals(vals);
    auto souts = new SArray<Val>(outs->data(), outs->size());
    SArray<int> *slens =
        lens ? new SArray<int>(lens->data(), lens->size()) : nullptr;
    int ts = ZPushPull(
        skeys, svals, souts, slens, cmd,
        [this, cb, souts, slens]() {
          delete souts;
          delete slens;
          if (cb)
            cb();
        },
        priority, psftype);
    return ts;
  }

  /**
   * \brief Waits until a push or pull has been finished
   *
   * Sample usage:
   * \code
   *   int ts = w.Pull(keys, &vals);
   *   Wait(ts);
   *   // now vals is ready for use
   * \endcode
   *
   * \param timestamp the timestamp returned by the push or pull
   */
  void Wait(int timestamp) { obj_->WaitRequest(timestamp); }

  /**
   * \brief zero-copy Push
   *
   * This function is similar to \ref Push except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPush(const SArray<Key> &keys, const SArray<Val> &vals,
            const SArray<int> &lens, int cmd = 0,
            const Callback &cb = nullptr, int priority = 0,
            PsfType psftype = PsfType::DensePush) {
    int ts = obj_->NewRequest(kServerGroup);
    AddCallback(ts, cb);
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.lens = lens;
    kvs.priority = priority;
    Send(ts, true, false, cmd, psftype, kvs);
    return ts;
  }

  /**
   * \brief zero-copy Pull
   *
   * This function is similar to \ref Pull except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPull(const SArray<Key> &keys, SArray<Val> *vals,
            SArray<int> *lens, int cmd = 0,
            const Callback &cb = nullptr, int priority = 0,
            PsfType psftype = PsfType::DensePull) {
    int ts = AddPullCB(keys, vals, lens, cmd, cb, psftype);
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.lens = *lens;
    kvs.priority = priority;
    Send(ts, false, true, cmd, psftype, kvs);
    return ts;
  }

  /**
   * \brief zero-copy PushPull
   *
   * This function is similar to \ref PushPull except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPushPull(const SArray<Key> &keys, const SArray<Val> &vals,
                SArray<Val> *outs, SArray<int> *lens, int cmd = 0,
                const Callback &cb = nullptr, int priority = 0,
                PsfType psftype = PsfType::PushPull) {
    int ts = AddPullCB(keys, outs, lens, cmd, cb, psftype);
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.priority = priority;
    if (lens)
      kvs.lens = *lens;
    Send(ts, true, true, cmd, psftype, kvs);
    return ts;
  }

private:
  /**
   * \brief internal pull, C/D can be either SArray or std::vector
   */
  template <typename C, typename D>
  int AddPullCB(const SArray<Key> &keys, C *vals, D *lens, int cmd,
                const Callback &cb, PsfType PsfType);

  /**
   * \brief add a callback for a request. threadsafe.
   * @param cb callback
   * @param timestamp the timestamp of the request
   */
  void AddCallback(int timestamp, const Callback &cb) {
    if (!cb)
      return;
    std::lock_guard<std::mutex> lk(mu_);
    callbacks_[timestamp] = cb;
  }

  /**
   * \brief run and delete the callback
   * \param timestamp the timestamp of the callback
   */
  void RunCallback(int timestamp);
  /**
   * \brief send the kv list to all servers
   * @param timestamp the timestamp of the request
   * @param push whether or not it is a push request
   * @param push whether or not it is a pull request
   * @param cmd command
   */
  void Send(int timestamp, bool push, bool pull, int cmd, PsfType psfType,
            const KVPairs<Val> &kvs);
  /** \brief internal receive handle */
  void Process(const Message &msg);
  
  /** \brief data buffer for received kvs for each timestamp */
  std::unordered_map<int, std::vector<KVPairs<Val>>> recv_kvs_;
  /** \brief callbacks for each timestamp */
  std::unordered_map<int, Callback> callbacks_;
  /** \brief lock */
  std::mutex mu_;
};

/**
 * \brief (1) We do the slicing operation in user code (both DL and ML part), thus
 * we remove the default slicer used in ps-lite. (2) One key in one request.
 */
template <typename Val>
void KVWorker<Val>::Send(int timestamp, bool push, bool pull, int cmd,
                         PsfType psftype, const KVPairs<Val> &kvs) {
  // find the target server
  const std::vector<Range> &server_range =
      Postoffice::Get()->GetServerKeyRanges();
  int server_num = (int)server_range.size();
  auto keys = kvs.keys;
  CHECK_LE(keys.size(), 2)
      << "the key size should be one/two(for binary ops) under our ps-mapping solution";
  obj_->AddResponse(timestamp, server_num - 1);
  Key k = keys[0];
  int server_id = 0;
  while (server_id < server_num && k >= server_range[server_id].begin()) {
    server_id++;
  }
  int target_server_id = server_id - 1;
  Message msg;
  msg.meta.app_id = obj_->app_id();
  msg.meta.customer_id = obj_->customer_id();
  msg.meta.request = true;
  msg.meta.push = push;
  msg.meta.pull = pull;
  msg.meta.head = cmd;
  msg.meta.timestamp = timestamp;
  msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
  msg.meta.priority = kvs.priority;
  msg.meta.psftype = psftype;

  if (kvs.keys.size()) {
    msg.AddData(kvs.keys);
    msg.AddData(kvs.vals);
    if (kvs.lens.size()) {
      msg.AddData(kvs.lens);
    }
  }
  Postoffice::Get()->van()->Send(msg);
}

/*
 * receive message from kv servers.
 */
template <typename Val> void KVWorker<Val>::Process(const Message &msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg);
    return;
  }
  // store the data for pulling
  int ts = msg.meta.timestamp;
  if (msg.meta.pull) {
    CHECK_GE(msg.data.size(), (size_t)2);
    KVPairs<Val> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];
    if (msg.data.size() > (size_t)2) {
      kvs.lens = msg.data[2];
    }

    mu_.lock();
    recv_kvs_[ts].push_back(kvs);
    mu_.unlock();
  }
  if (obj_->NumResponse(ts) == Postoffice::Get()->num_servers() - 1) {
    RunCallback(ts);
  }
}

template <typename Val> void KVWorker<Val>::RunCallback(int timestamp) {
  mu_.lock();
  auto it = callbacks_.find(timestamp);
  if (it != callbacks_.end()) {
    mu_.unlock();
    CHECK(it->second);
    it->second();
    mu_.lock();
    callbacks_.erase(it);
  }
  mu_.unlock();
}

/**
 * \brief Deal with the messages from many servers on one worker.
 *        In Athena case, there is server response for one request since we 
 *        do the slicing operation on worker side, i.e., the 
 *        recv_kvs[timestamp] contains exactly one KVPairs.
 */
template <typename Val>
template <typename C, typename D>
int KVWorker<Val>::AddPullCB(const SArray<Key> &keys, C *vals, D *lens, int cmd,
                             const Callback &cb, PsfType psftype) {
  int ts = obj_->NewRequest(kServerGroup);
  AddCallback(ts, [this, ts, keys, vals, lens, cb, psftype]() mutable {
    mu_.lock();
    auto &kvs = recv_kvs_[ts];
    mu_.unlock();

    CHECK_EQ(kvs.size(), 1) << "In Athena setting, we have one server response per request";
    auto& kv0 = kvs[0];
    auto& recver_vals = kv0.vals;

    if(vals->empty()){
      vals->resize(recver_vals.size(), 0);
    }
    else{
      CHECK_EQ(vals->size(), recver_vals.size()) << vals->size() << " " << recver_vals.size();;
    }
    Val *p_vals = vals->data();
    memcpy(p_vals, recver_vals.data(), recver_vals.size() * sizeof(Val));
    
    // lens is not needed. We only need the vals.
    // auto& recver_lens = kv0.lens;
    // if(lens->empty()){
    //   lens->resize(recver_lens.size(), 0);
    // }
    // else{
    //   CHECK_EQ(lens->size(), recver_lens.size()) << lens->size() << " " << recver_lens.size();
    // }
    // int *p_lens = lens->data();
    // memcpy(p_lens, recver_lens.data(), recver_lens.size() * sizeof(int));

    mu_.lock();
    recv_kvs_.erase(ts);
    mu_.unlock();
    if (cb)
      cb();
  });

  return ts;
}

} // namespace ps
#endif // PS_KV_WORKER_H_
