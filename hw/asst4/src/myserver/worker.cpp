#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <glog/logging.h>
#include <string>

#include "server/messages.h"
#include "server/worker.h"
#include "tools/cycle_timer.h"
#include "tools/work_queue.h"

using namespace std;

WorkQueue<Request_msg>* request_queue;
WorkQueue<Request_msg>* tellmenow_queue;
WorkQueue<Request_msg>* projectidea_queue;

bool is_special_node = false;

void* worker_thread(void*);
void* tellmenow_worker_thread(void*);
void* projectidea_worker_thread(void*);
inline void do_work(const Request_msg&);

void worker_node_init(const Request_msg& params) {
  int thread_num = 47;  // plus one project idea thread

  DLOG(INFO) << "**** Initializing worker: " << params.get_arg("tag") << " ****\n";

  int tag = stoi(params.get_arg("tag"));

  // special tellmenow thread on first node
  if (tag == 0) {
    pthread_t tellmenow_worker;
    pthread_create(&tellmenow_worker, NULL, tellmenow_worker_thread, NULL);
    pthread_detach(tellmenow_worker);

    // plus one special tellmenow thread
    thread_num = 46;
  }

  request_queue = new WorkQueue<Request_msg>;
  tellmenow_queue = new WorkQueue<Request_msg>;
  projectidea_queue = new WorkQueue<Request_msg>;

  // regular worker threads
  pthread_t workers[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    pthread_create(&workers[i], NULL, worker_thread, NULL);
    pthread_detach(workers[i]);
  }

  // special projectidea thread
  pthread_t projectidea_worker;
  pthread_create(&projectidea_worker, NULL, projectidea_worker_thread, NULL);
  pthread_detach(projectidea_worker);

}

void worker_handle_request(const Request_msg& req) {
  // Output debugging help to the logs (in a single worker node
  // configuration, this would be in the log logs/worker.INFO)
  DLOG(INFO) << "Worker got request: [" << req.get_tag() << ":" << req.get_request_string() << "]\n";

  string cmd = req.get_arg("cmd");

  // do not want tellme now and projectidea to be blocked by other requests
  if (cmd == "tellmenow") {  
    tellmenow_queue->put_work(req);
  } else if (cmd == "projectidea") {
    projectidea_queue->put_work(req);
  } else {
    request_queue->put_work(req);
  }
}

void* worker_thread(void* thread_args) {
  while (1) {
    Request_msg req = request_queue->get_work();
    do_work(req);
  }
  return NULL;
}

void* tellmenow_worker_thread(void* thread_args) {
  while (1) {
    Request_msg req = tellmenow_queue->get_work();
    do_work(req);
  }
  return NULL;
}

void* projectidea_worker_thread(void* thread_args) {
  while (1) {
    Request_msg req = projectidea_queue->get_work();
    do_work(req);
  }
  return NULL;
}

inline void do_work(const Request_msg& req) {
  Response_msg resp= req.get_tag();
  double startTime = CycleTimer::currentSeconds();
  execute_work(req, resp);
  double dt = CycleTimer::currentSeconds() - startTime;
  DLOG(INFO) << "Worker completed work in " << (1000.f * dt) << " ms (" << req.get_tag()  << ")\n";
  // send a response string to the master
  worker_send_response(resp);
}
