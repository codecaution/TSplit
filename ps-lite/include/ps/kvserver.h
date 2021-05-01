#ifndef PS_KV_SERVER_H_
#define PS_KV_SERVER_H_
#include "ps/base.h"
#include "ps/kvmeta.h"
#include "ps/kvpairs.h"
#include "ps/simple_app.h"
#include "ps/psf/PSFunc.h"
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
namespace ps {

/**
 * \brief A server node for maintaining key-value pairs
 */
template <typename Val> class KVServer : public SimpleApp {
public:
  /**
   * \brief constructor
   * \param app_id the app id, should match with \ref KVWorker's id
   */
  explicit KVServer(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    obj_ = new Customer(app_id, app_id,
                        std::bind(&KVServer<Val>::Process, this, _1));
  }

  /** \brief deconstructor */
  virtual ~KVServer() {
    delete obj_;
    obj_ = nullptr;
  }

  /**
   * \brief the handle to process a push/pull request from a worker
   * \param req_meta meta-info of this request
   * \param req_data kv pairs of this request
   * \param server this pointer
   */
  using ReqHandle = std::function<void(
      const KVMeta &req_meta, const KVPairs<Val> &req_data, KVServer *server)>;
  void set_request_handle(const ReqHandle &request_handle) {
    CHECK(request_handle) << "invalid request handle";
    request_handle_ = request_handle;
  }

  /**
   * \brief response to the push/pull request
   * \param req the meta-info of the request
   * \param res the kv pairs that will send back to the worker
   */
  void Response(const KVMeta &req, const KVPairs<Val> &res = KVPairs<Val>());

private:
  /** \brief internal receive handle */
  void Process(const Message &msg);
  /** \brief request handle */
  ReqHandle request_handle_;
};

/**
 * Process the message in parameter servers.
 * The data is [keys. vals, lens].
 */
template <typename Val> void KVServer<Val>::Process(const Message &msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg);
    return;
  }
  KVMeta meta;
  meta.cmd = msg.meta.head;
  meta.push = msg.meta.push;
  meta.pull = msg.meta.pull;
  meta.sender = msg.meta.sender;
  meta.timestamp = msg.meta.timestamp;
  meta.customer_id = msg.meta.customer_id;
  meta.psftype = msg.meta.psftype;

  KVPairs<Val> data;
  int n = msg.data.size();
  if (n) {
    CHECK_GE(n, 2);
    data.keys = msg.data[0];
    data.vals = msg.data[1];
    if (n == 3) {
      data.lens = msg.data[2];
    }
  }
  CHECK(request_handle_);
  request_handle_(meta, data, this);
}

template <typename Val>
void KVServer<Val>::Response(const KVMeta &req, const KVPairs<Val> &res) {
  Message msg;
  msg.meta.app_id = obj_->app_id();
  msg.meta.customer_id = req.customer_id;
  msg.meta.request = false;
  msg.meta.push = req.push;
  msg.meta.pull = req.pull;
  msg.meta.head = req.cmd;
  msg.meta.timestamp = req.timestamp;
  msg.meta.recver = req.sender;
  msg.meta.psftype = req.psftype;
  if (res.keys.size()) {
    msg.AddData(res.keys);
    msg.AddData(res.vals);
    if (res.lens.size()) {
      msg.AddData(res.lens);
    }
  }
  Postoffice::Get()->van()->Send(msg);
}

} // namespace ps
#endif // PS_KV_SERVER_H_
