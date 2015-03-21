#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <list>

#include "server/messages.h"
#include "server/master.h"

using namespace std;

static struct Master_state {

  // The mstate struct collects all the master node state into one
  // place.  You do not need to preserve any of the fields below, they
  // exist only to implement the basic functionality of the starter
  // code.

  bool server_ready;
  int max_num_workers;
  int num_pending_client_requests;
  int next_tag;

  // workers that are not processing requests
  queue<Worker_handle> free_workers;
  // workers that are processing reqeusts
  queue<Worker_handle> working_workers;
  // workers that has no processing power
  queue<Worker_handle> busy_workers;

  // all requests that need to be processed
  queue<Request_msg> request_queue;

  // key: request tag, value: client handle
  map<int, Client_handle> waiting_client;

} mstate;



void master_node_init(int max_workers, int& tick_period) {

  // set up tick handler to fire every 5 seconds. (feel free to
  // configure as you please)
  tick_period = 5;

  mstate.next_tag = 0;
  mstate.max_num_workers = max_workers;
  mstate.num_pending_client_requests = 0;

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

  int req_tag = resp.get_tag();
  Client_handle client = mstate.waiting_client.find(req_tag);
  if (client != mstate.waiting_client.end()) {
    send_client_response(client, resp);
  }

  mstate.num_pending_client_requests--;
  mstate.erase(req_tag);
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

  // The provided starter code cannot handle multiple pending client
  // requests.  The server returns an error message, and the checker
  // will mark the response as "incorrect"
  //if (mstate.num_pending_client_requests > 0) {
    //Response_msg resp(0);
    //resp.set_response("Oh no! This server cannot handle multiple outstanding requests!");
    //send_client_response(client_handle, resp);
    //return;
  //}

  // Save off the handle to the client that is expecting a response.
  // The master needs to do this it can response to this client later
  // when 'handle_worker_response' is called.
  mstate.waiting_client[client_req.get_tag()] = worker_handle;
  mstate.num_pending_client_requests++;

  // Fire off the request to the worker.  Eventually the worker will
  // respond, and your 'handle_worker_response' event handler will be
  // called to forward the worker's response back to the server.
  int tag = mstate.next_tag++;
  Request_msg worker_req(tag, client_req);
  // First assign requests to workers that is already processing requests. 
  if (!mstate.working_workers.empty()) {
    // remove from working queue
    Worker_handle worker = mstate.working_workers.pop();
    // send request
    send_request_to_worker(worker, worker_req);
    // add to busy workers
    mstate.busy_workers.push(worker);
  } else if (!mstate.free_workers.empty()){
    // remove from free queue
    Worker_handle worker = mstate.free_workers.pop();
    // send request
    send_request_to_worker(worker, worker_req);
    // add to working queue
    mstate.working_workers.push(worker);
  } else {  // add to request queues if no available workers
    mstate.request_queue.push(worker_req);
  }
  // We're done!  This event handler now returns, and the master
  // process calls another one of your handlers when action is
  // required.
}

void handle_tick() {

  // TODO: you may wish to take action here.  This method is called at
  // fixed time intervals, according to how you set 'tick_period' in
  // 'master_node_init'.

}
