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
#define PRINT_MESSAGE

using namespace std;

const int THREAD_NUM = 48;
const double THRESHOLD = 1.2;

typedef struct {
    int max_slots;
    int remaining_slots;
    int tag;
    bool processing_project_idea;
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

  int processing_project_idea_num;

  //int next_worker;

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
inline void worker_process_request(Worker_handle, Info&, const Request_msg&);

bool check_cache(Client_handle, const Request_msg&);
void start_new_worker();
void update_response_cache(int, const Response_msg&);
void process_request(const Request_msg&);
void process_compute_intensive_request(const Request_msg&);
int find_project_idea_worker();
void clear_queue();
void clear_compute_intensive_queue();
void clear_project_idea_queue();
void process_project_idea_request(const Request_msg&);

void master_node_init(int max_workers, int& tick_period) {
  // set up tick handler to fire every 1 seconds. 
  tick_period = 1;

  mstate.next_tag = 0;
  mstate.worker_num = 0;
  mstate.max_num_workers = max_workers;
  mstate.num_pending_client_requests = 0;
  
  //mstate.next_worker = 0;
  mstate.starting_worker = false;
  mstate.processing_project_idea_num = 0;

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
  info.processing_project_idea = false;

  mstate.worker_info[worker_handle] = info;
  mstate.workers.push_back(worker_handle);
  mstate.worker_num++;
  mstate.starting_worker = false;

#ifdef PRINT_MESSAGE
  DLOG(INFO) << "worker " << tag << " online! slots:" << info.max_slots << " total worker num: " << mstate.worker_num << endl;
#endif

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

#ifdef PRINT_MESSAGE
  DLOG(INFO) << "Master received a response from a worker: [" << resp.get_tag() << ":" << resp.get_response() << "]" << std::endl;
#endif

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
    DLOG(INFO) << "receive project idea response" << endl;
    info.processing_project_idea = false;
    mstate.processing_project_idea_num--;
    // try to fetch another project idea
    clear_project_idea_queue();
  }
  ++info.remaining_slots;
#ifdef DEBUG
  DLOG(INFO) << "add slot, worker " << info.tag<< " remaining slots: " << info.remaining_slots << endl;
#endif
  mstate.worker_info[worker_handle] = info;

  // try to clear queue
  clear_compute_intensive_queue();
}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

#ifdef PRINT_MESSAGE
  DLOG(INFO) << "Received request " << mstate.next_tag << ": " << client_req.get_request_string() << std::endl;
#endif

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

  // We're done!  This event handler now returns, and the master
  // process calls another one of your handlers when action is
  // required.
}

void process_request(const Request_msg& request_msg) {
  string cmd = request_msg.get_arg("cmd");

  // There is always a free slot to process tell me now in worker[0]
  DLOG(INFO) << "cmd: " << cmd << endl;
  if (cmd.compare("tellmenow") == 0) {
    Worker_handle worker_handle = mstate.workers[0];
    Info info = get_worker_info(worker_handle);
    worker_process_request(worker_handle, info, request_msg);
  } else if (cmd.compare("projectidea") == 0) {
    DLOG(INFO) << "process project idea" << endl;
    process_project_idea_request(request_msg);
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
#ifdef DEBUG
  DLOG(INFO) << "add request " << request_msg.get_tag() << "to queue, size: " << mstate.compute_intensive_queue.size() << endl;
#endif
  // ask for a new node
  if (!mstate.starting_worker) {
    start_new_worker();
#ifdef PRINT_MESSAGE
    DLOG(INFO) << "Starting new worker now" << endl;
#endif
  }
}

void process_project_idea_request(const Request_msg& request_msg) {
  // first try to assign to a worker 
  for (int i = 0; i < mstate.worker_num; ++i) {
    Worker_handle worker_handle = mstate.workers[i];
    Info info = get_worker_info(worker_handle);
    if (!info.processing_project_idea) {
      info.processing_project_idea = true;
      mstate.processing_project_idea_num++;
      worker_process_request(worker_handle, info, request_msg); 
#ifdef DEBUG 
      DLOG(INFO) << "send project idea request to worker " << info.tag << endl;
#endif

      return;
    } 
  }

  // reach here if no slots
  mstate.project_idea_queue.push(request_msg);
#ifdef DEBUG
  DLOG(INFO) << "send project idea request " << request_msg.get_tag() << " to queue, size: " << mstate.project_idea_queue.size() << endl;
#endif
  if (!mstate.starting_worker) {
    start_new_worker();
#ifdef PRINT_MESSAGE
    DLOG(INFO) << "Starting new worker now" << endl;
#endif
  }
}

void clear_queue() {
  if (!mstate.project_idea_queue.empty()) {
    DLOG(INFO) << "calling clear project idea queue function" << endl;
    clear_project_idea_queue();
  }
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

void clear_project_idea_queue() {
  for (int i = 0; i < mstate.worker_num; ++i) {
    if (mstate.project_idea_queue.empty()) {
        break;
    }
    Worker_handle worker_handle = mstate.workers[i];
    Info info = get_worker_info(worker_handle);
    if (!info.processing_project_idea) {
      Request_msg request_msg = mstate.project_idea_queue.front();
      mstate.project_idea_queue.pop();
      info.processing_project_idea = true;
      worker_process_request(worker_handle, info, request_msg);
    }
  }
}

/*
 * @brief Process request by the specific worker
 * Return the number of pending request
 */
inline void worker_process_request(Worker_handle worker_handle, 
        Info& info, const Request_msg& worker_req) {
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
    DLOG(ERROR) << "Cannot find tag" << endl;
  }
}

/*
 * @brief we want to only keep one spare project idea worker each time
 */
void kill_worker() {
  vector<Worker_handle>::iterator it = mstate.workers.begin() + 1;
  while (it != mstate.workers.end()) {
    // we want to spare one node to process project idea request
    if (mstate.worker_num <= mstate.processing_project_idea_num + 1) {
      break;
    }
    Worker_handle worker_handle = *it;
    Info info = get_worker_info(worker_handle);
#ifdef DEBUG
    DLOG(INFO) << "worker tag: " << info.tag << " remaining_slots: " << info.remaining_slots << endl;
#endif
    if (info.remaining_slots == info.max_slots) {
      it = mstate.workers.erase(it);
      mstate.worker_info.erase(worker_handle);
      kill_worker_node(worker_handle);
      mstate.worker_num--;
      DLOG(INFO) << "KILL worker " << info.tag <<  "!" << 
          "project idea num: " << mstate.processing_project_idea_num << endl;
    } else {
      ++it;
    }
  }
}

void handle_tick() {
#ifdef DEBUG
  DLOG(INFO) << "Queue length: " << mstate.compute_intensive_queue.size() << endl;
#endif
  // clear queue first
  clear_queue();

  // add node if 
  // 1. not reach maximum allowed workers
  // 2. there is a queue of compute intensive requests
  // 3. there is a queue of project idea requests
  // 4. no spare node to process project idea requests
  if (mstate.worker_num < mstate.max_num_workers 
          && (!mstate.compute_intensive_queue.empty() 
          || !mstate.project_idea_queue.empty()
          || mstate.processing_project_idea_num == mstate.worker_num)) {
    start_new_worker();
  }
  
  if (mstate.workers.empty()) {
    DLOG(INFO) << "no worker yet" << endl;
    return;
  }

  // kill worker if too many
  kill_worker();
  /*
  // kill node if its is free
  vector<Worker_handle>::iterator it = mstate.workers.begin() + 1;
  while (it != mstate.workers.end()) {
    Worker_handle worker_handle = *it;
    Info info = get_worker_info(worker_handle);
#ifdef DEBUG
    DLOG(INFO) << "worker tag: " << info.tag << " remaining_slots: " << info.remaining_slots << endl;
#endif
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
  */
}

inline Client_handle get_client_handle(int tag) {
  map<int,Client_handle>::iterator client_it = mstate.waiting_client.find(tag);
#ifdef DEBUG
  if (client_it == mstate.waiting_client.end()) {
    DLOG(ERROR) << "Cannot find client" << endl;
  }
#endif
  return client_it->second;
}

inline Info get_worker_info(Worker_handle worker_handle) {
  map<Worker_handle,Info>::iterator info_it 
      = mstate.worker_info.find(worker_handle);
#ifdef DEBUG
  if (info_it == mstate.worker_info.end()) {
    DLOG(ERROR) << "CANNOT FIND INFO" << endl;
  }
#endif
  Info info = info_it->second;
  return info;
}

