#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <queue>
#include <vector>
#include <iostream>
#include <climits>

#include "server/messages.h"
#include "server/master.h"

#define DEBUG

using namespace std;

const int THREAD_NUM = 24;
const double THRESHOLD = 1.5;

typedef struct {
    int max_slots;
    int remaining_slots;
    int tag;
    int processing_project_idea;
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

  int next_worker;

  // there is some worker booting now
  bool starting_worker;

  // workers
  vector<Worker_handle> workers;
  // key: worker handle, value: worker infomation
  map<Worker_handle, Info> worker_info;

  // key: request tag, value: client handle
  map<int, Client_handle> waiting_client;

  // key: request tag, value: request string 
  map<int, string> request_map;
  // request cache, key: request string, value: response msg
  map<string, Response_msg> request_cache;

  // project idea queue
  queue<Request_msg> project_idea_queue;
  // compute intensive queue
  queue<Request_msg> compute_intensive_queue;
} mstate;

inline Info get_worker_info(Worker_handle);
inline Client_handle get_client_handle(int tag);

bool check_cache(Client_handle, const Request_msg&);
void start_new_worker();
void update_response_cache(int, const Response_msg&);
void process_request(const Request_msg&);
void worker_process_request(Worker_handle, Info, const Request_msg&);
void process_compute_intensive_request(const Request_msg&);
int find_project_idea_worker();
void clear_queue();
void clear_compute_intensive_queue();

void master_node_init(int max_workers, int& tick_period) {

  // set up tick handler to fire every 1 seconds. 
  tick_period = 1;

  mstate.next_tag = 0;
  mstate.worker_num = 0;
  mstate.max_num_workers = max_workers;
  mstate.num_pending_client_requests = 0;
  
  mstate.next_worker = 0;
  mstate.starting_worker = false;

  // don't mark the server as ready until the server is ready to go.
  // This is actually when the first worker is up and running, not
  // when 'master_node_init' returnes
  mstate.server_ready = false;

  start_new_worker();
}

/*
 * Start a new worker node
 */
void start_new_worker() {
  if (!mstate.starting_worker 
          && mstate.worker_num < mstate.max_num_workers) {
    int tag = mstate.next_tag++;
    Request_msg req(tag);
    req.set_arg("name", "my worker " + tag);
    mstate.starting_worker = true;
    request_new_worker_node(req);
  }
}

/* 
 * 'tag' allows you to identify which worker request 
 * this response corresponds to. 
 *
 * Each a worker goes online, check if there is pending requests
 */
void handle_new_worker_online(Worker_handle worker_handle, int tag) {
  Info info;
  
  if (tag == 0) {  // set first worker as special one
    info.max_slots = THREAD_NUM - 1;
  } else {
    info.max_slots = static_cast<int>(THREAD_NUM * THRESHOLD);
  }
  info.remaining_slots = info.max_slots;
  info.tag = tag;
  info.processing_project_idea = -1;

  mstate.worker_info[worker_handle] = info;
  mstate.workers.push_back(worker_handle);
  mstate.worker_num++;
  mstate.starting_worker = false;

  DLOG(INFO) << "worker " << tag << " online! slots:" << info.max_slots << endl;

  // Now that a worker is booted, let the system know the server is
  // ready to begin handling client requests.  The test harness will
  // now start its timers and start hitting your server with requests.
  if (mstate.server_ready == false) {
    server_init_complete();
    mstate.server_ready = true;
  }

  // try to clear queue each time come online
  clear_queue();
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
  map<int, string>::iterator request_it = mstate.request_map.find(resp_tag);
  string req_str;
  if (request_it != mstate.request_map.end()) {
    req_str = request_it->second;
  }
  if (req_str.find("projectidea") != string::npos) {
    info.processing_project_idea = -1;
  }
  ++info.remaining_slots;
  DLOG(INFO) << "add slot, worker " << info.tag<< " remaining slots: " << info.remaining_slots << endl;
  mstate.worker_info[worker_handle] = info;
}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

  DLOG(INFO) << "Received request " << mstate.next_tag << ": " << client_req.get_request_string() << std::endl;

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
  
  process_request(request_msg);

  /*
  string cmd = client_req.get_arg("cmd");
  Worker_handle worker_handle;
  if (cmd.compare("projectidea") == 0) {
    int index = find_project_idea_worker();
    worker_handle = mstate.workers[index];
    Info info = get_worker_info(worker_handle);
    info.processing_project_idea = tag;
    mstate.worker_info[worker_handle] = info;
  } else {
    worker_handle = mstate.workers[mstate.next_worker++ % mstate.worker_num];
  }
  send_request_to_worker(worker_handle, request_msg);
  */

  // We're done!  This event handler now returns, and the master
  // process calls another one of your handlers when action is
  // required.
}

void process_request(const Request_msg& request_msg) {
  string cmd = request_msg.get_arg("cmd");

  // There is always a free slot to process tell me now in worker[0]
  if (cmd.compare("tellmenow") == 0) {
    Worker_handle worker_handle = mstate.workers[0];
    Info info = get_worker_info(worker_handle);
    worker_process_request(worker_handle, info, request_msg);
  } else if (cmd.compare("project_idea") == 0) {
    // TODO
  } else if (cmd.compare("bandwidth") == 0) {
    // TODO
  } else if (cmd.compare("compareprimes") == 0) {
    // TODO
  } else {  // compute intensive
    process_compute_intensive_request(request_msg);
  }
}

void process_compute_intensive_request(const Request_msg& request_msg) {
  for (int i = 0; i < mstate.worker_num; ++i) {
    Worker_handle worker_handle = mstate.workers[i];
    Info info = get_worker_info(worker_handle);

    if (info.remaining_slots > 0) {
      worker_process_request(worker_handle, info, request_msg); 
      return;
    }
  }
  // reach here if no slots
  mstate.compute_intensive_queue.push(request_msg);
  DLOG(INFO) << "add request " << request_msg.get_tag() << "to queue, size: " << mstate.compute_intensive_queue.size() << endl;
  // ask for a new node
  if (!mstate.starting_worker) {
    start_new_worker();
    DLOG(INFO) << "Starting new worker now" << endl;
  }
}

void clear_queue() {
  // TODO try to clear project idea first
  
  if (!mstate.compute_intensive_queue.empty()) {
    clear_compute_intensive_queue();
  }
}

void clear_compute_intensive_queue() {
  for (int i = 0; i < mstate.worker_num; ++i) {
    if (mstate.compute_intensive_queue.empty()) {
        break;
    }
    Worker_handle worker_handle = mstate.workers[i];
    Info info = get_worker_info(worker_handle);
    while (!mstate.compute_intensive_queue.empty() &&
            info.remaining_slots > 0) {
      Request_msg request_msg = mstate.compute_intensive_queue.front();
      mstate.compute_intensive_queue.pop();
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
  info.remaining_slots--;
  mstate.worker_info[worker_handle] = info;

#ifdef DEBUG
    DLOG(INFO) << "send request " << worker_req.get_tag() << " to worker " << info.tag << " remaining slots: " << info.remaining_slots << endl;
#endif
}

bool check_cache(Client_handle client_handle, const Request_msg& client_req) {
  map<string, Response_msg>::iterator request_it = mstate.request_cache.find(client_req.get_request_string());
  if (request_it != mstate.request_cache.end()) {
    Response_msg resp = request_it->second;
    // reset tag number
    resp.set_tag(mstate.next_tag++);
    send_client_response(client_handle, resp);
#ifdef DEBUG
    DLOG(INFO) << "Cache hit: " << resp.get_tag() << std::endl;
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
}

void handle_tick() {
  DLOG(INFO) << "Queue length: " << mstate.compute_intensive_queue.size() << endl;
  // clear queue first
  clear_queue();

  // add node if needed
  if (mstate.worker_num < mstate.max_num_workers 
          && (!mstate.compute_intensive_queue.empty() 
          || !mstate.project_idea_queue.empty())) {
    start_new_worker();
  }
  
  // kill node if its is free
  vector<Worker_handle>::iterator it = mstate.workers.begin() + 1;
  while (it != mstate.workers.end()) {
    Worker_handle worker_handle = *it;
    Info info = get_worker_info(worker_handle);
    if (info.remaining_slots == info.max_slots) {
      it = mstate.workers.erase(it);
      mstate.worker_info.erase(worker_handle);
      kill_worker_node(worker_handle);
      mstate.worker_num--;
      DLOG(INFO) << "KILL worker " << info.tag <<  "!" << endl;
    } else {
      ++it;
    }
  }
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
#ifdef DEBUG
  if (info_it == mstate.worker_info.end()) {
    DLOG(INFO) << "CANNOT FIND INFO" << endl;
  }
#endif
  Info info = info_it->second;
  return info;
}

int find_project_idea_worker() {
  int min_tag = INT_MAX;
  int min_index = 0;
  for (int i = 0; i < mstate.worker_num; ++i) {
    Worker_handle worker_handle = mstate.workers[i];
    Info info = get_worker_info(worker_handle);
    if (info.processing_project_idea == -1) {
      cout << "index: " << i << endl;
      return i;
      //min_index = i;
      //break;
    } else if (info.processing_project_idea < min_tag) {
      min_tag = info.processing_project_idea;
      min_index = i;
    }
  }
  cout << "SHITTTTTT: " << min_index << endl;
  return min_index;
}
