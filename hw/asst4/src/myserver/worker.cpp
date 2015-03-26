
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <glog/logging.h>

#include "server/messages.h"
#include "server/worker.h"
#include "tools/cycle_timer.h"
#include "tools/work_queue.h"

const int THREAD_NUM = 48; 
WorkQueue<Request_msg>* request_queue;

void* worker_thread(void* thread_args);


void worker_node_init(const Request_msg& params) {
  DLOG(INFO) << "**** Initializing worker: " << params.get_arg("name") << " ****\n";

  request_queue = new WorkQueue<Request_msg>;
  pthread_t workers[THREAD_NUM];

  for (int i = 0; i < THREAD_NUM; ++i) {
    pthread_create(&workers[i], NULL, worker_thread, NULL);
    pthread_detach(workers[i]);
  }
}

void worker_handle_request(const Request_msg& req) {
  // Output debugging help to the logs (in a single worker node
  // configuration, this would be in the log logs/worker.INFO)
  DLOG(INFO) << "Worker got request: [" << req.get_tag() << ":" << req.get_request_string() << "]\n";
  // simple put into queue
  request_queue->put_work(req);
}

void* worker_thread(void* thread_args) {
  while (1) {
    Request_msg req = request_queue->get_work();
    Response_msg resp= req.get_tag();
    
    double startTime = CycleTimer::currentSeconds();
    
    execute_work(req, resp);
    double dt = CycleTimer::currentSeconds() - startTime;
    DLOG(INFO) << "Worker completed work in " << (1000.f * dt) << " ms (" << req.get_tag()  << ")\n";
    // send a response string to the master
    worker_send_response(resp);
  }
  return NULL;
}
