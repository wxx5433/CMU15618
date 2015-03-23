#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <queue>
#include <list>

#include "server/messages.h"
#include "server/master.h"

//#define DEBUG

using namespace std;

const int THREAD_NUM = 24;

typedef struct {
    int remaining_slots;
    int tag;
} Info;

static struct Master_state {

  // The mstate struct collects all the master node state into one
  // place.  You do not need to preserve any of the fields below, they
  // exist only to implement the basic functionality of the starter
  // code.

  bool server_ready;
  int worker_num;
  int max_num_workers;
  int num_pending_client_requests;
  int next_tag;

  // workers
  list<Worker_handle> workers;
  // key: worker handle, value: worker infomation
  map<Worker_handle, Info> worker_info;

  // key: request tag, value: client handle
  map<int, Client_handle> waiting_client;

  // key: request tag, value: request string 
  map<int, string> request_map;
  // request cache, key: request string, value: response msg
  map<string, Response_msg> request_cache;

} mstate;

inline bool isFreeWorker(Info info);
inline int get_remaining_power();
inline Info get_worker_info(Worker_handle);
inline Client_handle get_client_handle(int tag);

bool check_cache(Client_handle, const Request_msg&);
void start_new_worker();
void update_response_cache(int, const Response_msg&);
void worker_process_request(Worker_handle, Info, const Request_msg&);
void process_request(const Request_msg&);

void master_node_init(int max_workers, int& tick_period) {

  // set up tick handler to fire every 5 seconds. (feel free to
  // configure as you please)
  tick_period = 2;

  mstate.next_tag = 0;
  mstate.worker_num = 0;
  mstate.max_num_workers = max_workers;
  mstate.num_pending_client_requests = 0;

  // don't mark the server as ready until the server is ready to go.
  // This is actually when the first worker is up and running, not
  // when 'master_node_init' returnes
  mstate.server_ready = false;

  start_new_worker();
}

/* 
 * 'tag' allows you to identify which worker request 
 * this response corresponds to. 
 */
void handle_new_worker_online(Worker_handle worker_handle, int tag) {
  // add the new worker to free workers queue
  Info info;
  info.remaining_slots = THREAD_NUM;
  info.tag = tag;
  mstate.worker_info[worker_handle] = info;
  mstate.workers.push_back(worker_handle);
  mstate.worker_num++;

  DLOG(INFO) << "worker " << tag << " online!" << endl;

  // Now that a worker is booted, let the system know the server is
  // ready to begin handling client requests.  The test harness will
  // now start its timers and start hitting your server with requests.
  if (mstate.server_ready == false) {
    server_init_complete();
    mstate.server_ready = true;
  }
}

void handle_worker_response(Worker_handle worker_handle, const Response_msg& resp) {

  // Master node has received a response from one of its workers.
  // Here we directly return this response to the client.

  DLOG(INFO) << "Master received a response from a worker: [" << resp.get_tag() << ":" << resp.get_response() << "]" << std::endl;

  // send response to client
  int resp_tag = resp.get_tag();
  Client_handle client_handle = get_client_handle(resp_tag);
  send_client_response(client_handle, resp);
  mstate.num_pending_client_requests--;

  // add response message to cache
  update_response_cache(resp_tag, resp);

  // update worker info
  Info info = get_worker_info(worker_handle);
  ++info.remaining_slots;
  mstate.worker_info[worker_handle] = info;

}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

  DLOG(INFO) << "Received request: " << client_req.get_request_string() << std::endl;

  // check cache
  if (check_cache(client_handle, client_req)) {
    return;
  }

  // You can assume that traces end with this special message.  It
  // exists because it might be useful for debugging to dump
  // information about the entire run here: statistics, etc.
  if (client_req.get_arg("cmd") == "lastrequest") {
    Response_msg resp(0);
    resp.set_response("ack");
    send_client_response(client_handle, resp);
    return;
  }

  // Save off the handle to the client that is expecting a response.
  // The master needs to do this it can response to this client later
  // when 'handle_worker_response' is called.
  int tag = mstate.next_tag++;
  mstate.waiting_client[tag] = client_handle;
  mstate.request_map[tag] = client_req.get_request_string();
  mstate.num_pending_client_requests++;

  // Fire off the request to the worker.  Eventually the worker will
  // respond, and your 'handle_worker_response' event handler will be
  // called to forward the worker's response back to the server.
  Request_msg request_msg(tag, client_req);
  Worker_handle worker_handle = mstate.workers.front();
  send_request_to_worker(worker_handle, request_msg);

  //process_request(request_msg);
  // We're done!  This event handler now returns, and the master
  // process calls another one of your handlers when action is
  // required.
}

void process_request(const Request_msg& request_msg) {
  int free_worker_num = 0;

  // first try to assign requests to workers already processing some requests
  list<Worker_handle>::iterator it;
  for (it = mstate.workers.begin(); it != mstate.workers.end(); ++it) {
    Worker_handle worker_handle = *it;
    Info info = get_worker_info(worker_handle);
    // ignore free workers first
    if (info.remaining_slots == THREAD_NUM) {
      ++free_worker_num;
      continue;
    }
    worker_process_request(worker_handle, info, request_msg); 
  }

  // reach here if there are still requests
  if (free_worker_num != 0) {
    for (it = mstate.workers.begin(); it != mstate.workers.end(); ++it) {
      Worker_handle worker_handle = *it;
      Info info = get_worker_info(worker_handle);

      worker_process_request(worker_handle, info, request_msg);
    }
  }
}

/*
 * @brief Process request by the specific worker
 * Return the number of pending request
 */
void worker_process_request(Worker_handle worker_handle, 
        Info info, const Request_msg& worker_req) {
  // send request
  send_request_to_worker(worker_handle, worker_req);
#ifdef DEBUG
    DLOG(INFO) << "worker " << info.tag << " remaining slots: " << info.remaining_slots << endl;
#endif

  --info.remaining_slots;
  mstate.worker_info[worker_handle] = info;
}

bool check_cache(Client_handle client_handle, const Request_msg& client_req) {
  map<string, Response_msg>::iterator request_it = mstate.request_cache.find(client_req.get_request_string());
  if (request_it != mstate.request_cache.end()) {
    Response_msg resp = request_it->second;
    // reset tag number
    resp.set_tag(mstate.next_tag++);
    send_client_response(client_handle, resp);
#ifdef DEBUG
    DLOG(INFO) << "Cache hit: " << client_req.get_request_string() << std::endl;
#endif
    return true;
  }
  return false; 
}

void update_response_cache(int resp_tag, const Response_msg& resp) {
  map<int, string>::iterator request_it = mstate.request_map.find(resp_tag);
  if (request_it != mstate.request_map.end()) {
    string request_string = request_it->second;
    mstate.request_cache[request_string] = resp;
  } else {
    DLOG(INFO) << "Cannot find tag" << endl;
  }
  mstate.request_map.erase(resp_tag);
}

void start_new_worker() {
  if (mstate.worker_num <= mstate.max_num_workers) {
    int tag = mstate.next_tag++;
    Request_msg req(tag);
    req.set_arg("name", "my worker " + tag);
    request_new_worker_node(req);
    mstate.worker_num++;
  }
}

void handle_tick() {

  // TODO: you may wish to take action here.  This method is called at
  // fixed time intervals, according to how you set 'tick_period' in
  // 'master_node_init'.
  /*
  int remaining_power = get_remaining_power();
  if (!mstate.request_queue.empty() && 
          remaining_power <= THREAD_NUM / 2) {
    int need_worker_num = (mstate.request_queue.size()
            + THREAD_NUM - 1) / THREAD_NUM;
    need_worker_num = max(1, need_worker_num);

    for (int i = 0; i < need_worker_num; ++i) {
      start_new_worker();
    }
  } else { 
    list<Worker_handle>::iterator it = mstate.workers.begin(); 
    while (it != mstate.workers.end() && mstate.worker_num > 2
            && get_remaining_power() > THREAD_NUM / 2 ) {
      Worker_handle worker_handle = *it;
      Info info = get_worker_info(worker_handle);
      if (isFreeWorker(info)) {
        it = mstate.workers.erase(it);
        mstate.worker_num--;
        mstate.worker_info.erase(worker_handle);
        kill_worker_node(worker_handle);
        DLOG(INFO) << "KILL worker!" << endl;
      } else {
        ++it;
      }
    }
  }
  */
}

inline Client_handle get_client_handle(int tag) {
  map<int,Client_handle>::iterator client_it = mstate.waiting_client.find(tag);
#ifdef DEBUG
  if (client_it == mstate.waiting_client.end()) {
    DLOG(INFO) << "Cannot find client" << endl;
  }
#endif
  return client_it->second;
}

inline Info get_worker_info(Worker_handle worker_handle) {
  map<Worker_handle,Info>::iterator info_it 
      = mstate.worker_info.find(worker_handle);
  Info info = info_it->second;
  return info;
}

inline bool isFreeWorker(Info info) {
  if (info.remaining_slots == THREAD_NUM) {
    return true;
  }
  return false;
}

inline int get_remaining_power() {
  return mstate.worker_num * THREAD_NUM 
      - mstate.num_pending_client_requests;
}
