#ifndef PS_KV_META_H_
#define PS_KV_META_H_
namespace ps {

/** \brief meta information about a kv request */
struct KVMeta {
  /** \brief the int cmd */
  int cmd;
  /** \brief whether or not this is a push request */
  bool push;
  /** \brief whether or not this is a pull request */
  bool pull;
  /** \brief sender's node id */
  int sender;
  /** \brief the associated timestamp */
  int timestamp;
  /** \brief the customer id of worker */
  int customer_id;
  /** \brief server-side computation op for keys */
  PsfType psftype;
};

}// namespace ps
#endif  // PS_KV_META_H_