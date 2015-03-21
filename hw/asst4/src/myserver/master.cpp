#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <list>
#include <queue>
#include <iostream>

#include "server/messages.h"
#include "server/master.h"

//#define DEBUG

using namespace std;

enum State {FREE, WORKING, BUSY};
const int WORKER_NUM = 24;

typedef struct {
    State state;
    int processing_request_num;
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

  // workers that are not processing requests
  queue<Worker_handle> free_workers;
  // workers that are processing reqeusts
  list<Worker_handle> working_workers;
  // workers that has no processing power
  queue<Worker_handle> busy_workers;

  // all requests that need to be processed
  queue<Request_msg> request_queue;

  map<Worker_handle, Info> worker_info;
  // key: request tag, value: client handle
  map<int, Client_handle> waiting_client;

} mstate;



void master_node_init(int max_workers, int& tick_period) {

  // set up tick handler to fire every 5 seconds. (feel free to
  // configure as you please)
  tick_period = 2;

  mstate.next_tag = 0;
  mstate.worker_num = 0;
  mstate.max_num_workers = max_workers;
  mstate.num_pending_client_requests = 0;

  cout << "max workers num: " << mstate.max_num_workers << endl;

  // don't mark the server as ready until the server is ready to go.
  // This is actually when the first worker is up and running, not
  // when 'master_node_init' returnes
  mstate.server_ready = false;

  // fire off a request for a new worker
  int tag = mstate.next_tag++;
  Request_msg req(tag);
  req.set_arg("name", "my worker " + tag);
  request_new_worker_node(req);
}

/* 
 * 'tag' allows you to identify which worker request 
 * this response corresponds to. 
 */
void handle_new_worker_online(Worker_handle worker_handle, int tag) {
  // add the new worker to free workers queue
  mstate.free_workers.push(worker_handle);
  Info info;
  info.state = FREE;
  info.processing_request_num = 0;
  info.tag = tag;
  mstate.worker_info[worker_handle] = info;
  mstate.free_workers.push(worker_handle);
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

  int resp_tag = resp.get_tag();
#ifdef DEBUG
  DLOG(INFO) << "resp tag: " << resp_tag << endl;
  for (std::map<int,Client_handle>::iterator it=mstate.waiting_client.begin(); 
          it!=mstate.waiting_client.end(); ++it) {
    DLOG(INFO) << it->first << " => " << it->second << endl;
  }
#endif
  map<int,Client_handle>::iterator client_it = mstate.waiting_client.find(resp_tag);
  if (client_it != mstate.waiting_client.end()) {
    send_client_response(client_it->second, resp);
  } else {
    DLOG(INFO) << "Cannot find client" << endl;
  }

  map<Worker_handle,Info>::iterator info_it = mstate.worker_info.find(worker_handle);
  if (info_it != mstate.worker_info.end()) {
    Info info = info_it->second;
    info.processing_request_num--;
    if (info.processing_request_num == 0) {
      info.state = FREE;
      mstate.working_workers.remove(worker_handle);
      // add to free workers list
      mstate.free_workers.push(worker_handle);
    }
    mstate.worker_info[worker_handle] = info;
  } else {
    DLOG(INFO) << "Cannot find worker info" << endl;
  }

  mstate.num_pending_client_requests--;

  // if request queue is not empty, fetch one more
  if (!mstate.request_queue.empty()) {
    Request_msg worker_req = mstate.request_queue.front();
    mstate.request_queue.pop();
    send_request_to_worker(worker_handle, worker_req);
    map<Worker_handle,Info>::iterator info_it 
        = mstate.worker_info.find(worker_handle);
    Info info = info_it->second;
    DLOG(INFO) << "send a request to worker " << info.tag << ", request num: " << info.processing_request_num << endl;
  }
}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

  DLOG(INFO) << "Received request: " << client_req.get_request_string() << std::endl;

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
#ifdef DEBUG
  DLOG(INFO) << "add request" << endl;
  for (std::map<int,Client_handle>::iterator it=mstate.waiting_client.begin(); 
          it!=mstate.waiting_client.end(); ++it) {
    DLOG(INFO) << it->first << " => " << it->second << endl;
  }
#endif
  mstate.num_pending_client_requests++;

  // Fire off the request to the worker.  Eventually the worker will
  // respond, and your 'handle_worker_response' event handler will be
  // called to forward the worker's response back to the server.
  Request_msg worker_req(tag, client_req);

  // First assign requests to workers that is already processing requests. 
  map<Worker_handle,Info>::iterator info_it;
  if (!mstate.working_workers.empty()) {
    Worker_handle worker = mstate.working_workers.front();
    // update info
    info_it = mstate.worker_info.find(worker);
#ifdef DEBUG
    if (info_it == mstate.worker_info.end()) {
      DLOG(INFO) << "Cannot find worker info" << endl;
    }
#endif
    Info info = info_it->second;
    info.processing_request_num++;
    // if the worker has max processing requests
    if (info.processing_request_num == WORKER_NUM) {
      info.state = BUSY;
      // add to busy workers
      mstate.working_workers.pop_front();
      mstate.busy_workers.push(worker);
    }
    mstate.worker_info[worker] = info;
    // send request
    send_request_to_worker(worker, worker_req);

    DLOG(INFO) << "send a request to worker " << info.tag << ", request num: " << info.processing_request_num << endl;

  } else if (!mstate.free_workers.empty()){
    // get one worker from free lists
    Worker_handle worker = mstate.free_workers.front();
    mstate.free_workers.pop();
    // update info
    info_it = mstate.worker_info.find(worker);
#ifdef DEBUG
    if (info_it == mstate.worker_info.end()) {
      DLOG(INFO) << "Cannot find worker info" << endl;
    }
#endif
    Info info = info_it->second;
    info.processing_request_num++;
    info.state = WORKING;
    mstate.worker_info[worker] = info;
    // add to working workers list
    mstate.working_workers.push_back(worker);
    // send request
    send_request_to_worker(worker, worker_req);

    DLOG(INFO) << "send a request to worker " << info.tag << ", request num: " << info.processing_request_num << endl;

  } else {  // add to request queues if no available workers
    mstate.request_queue.push(worker_req);
    DLOG(INFO) << "add request into master queue" << endl;
  }
  // We're done!  This event handler now returns, and the master
  // process calls another one of your handlers when action is
  // required.
}

void handle_tick() {

  // TODO: you may wish to take action here.  This method is called at
  // fixed time intervals, according to how you set 'tick_period' in
  // 'master_node_init'.
  if (mstate.worker_num < mstate.max_num_workers) {
    int tag = mstate.next_tag++;
    Request_msg req(tag);
    req.set_arg("name", "my worker " + tag);
    request_new_worker_node(req);
    mstate.worker_num++;
  }
}
