#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "CycleTimer.h"
#include "bfs.h"
#include "graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->alloc_count = count;
    list->present = (int*)malloc(sizeof(int) * list->alloc_count);
    vertex_set_clear(list);
}

int bottom_up_step(graph* g, unsigned char* frontier, unsigned char* new_frontier, int* distances, int step) {
#ifdef DEBUG
    printf("bottom up\n");
#endif
    int count = 0;
    #pragma omp parallel for schedule(dynamic, 400) reduction(+:count)
    for (int node = 0; node < g->num_nodes; ++node) {  
        // already find distances
        if (distances[node] == NOT_VISITED_MARKER) {
         
          int private_count = 0;
          int start_edge = g->incoming_starts[node];
          int end_edge = (node == g->num_nodes - 1) ? g->num_edges: g->outgoing_starts[node + 1];
           
          for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            int index = incoming / 8;
            int offset = incoming % 8;
            // nodes just find in last step
            if (frontier[index] & (1 << offset)) { 
                distances[node] = step + 1;
                index = node / 8;
                offset = node % 8;
                new_frontier[index] |= (1 << offset);
                ++private_count;
                break;
            }
        }
        count += private_count;
      }
     
    }
    return count;
}

void bfs_bottom_up(graph* graph, solution* sol)
{
    int bitmaplen = graph->num_nodes / 8 + 1;
    unsigned char* bitfrontier = (unsigned char*)malloc(sizeof(char) * bitmaplen);
    unsigned char* bitnewfrontier = (unsigned char*)malloc(sizeof(char) * bitmaplen);
 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bitmaplen; i++) {
      bitfrontier[i] &= 0;
      bitnewfrontier[i] &= 0;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    bitfrontier[ROOT_NODE_ID] |= 1;
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    int count = 1;
    while (count != 0) {
#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        count = bottom_up_step(graph, bitfrontier, bitnewfrontier, sol->distances, step++);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif
    }
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
int top_down_step(graph* g, unsigned char* frontier, unsigned char* new_frontier, int* distances, int step) {
#ifdef DEBUG
    printf("top down\n");
#endif
    int count = 0;
    #pragma omp parallel for schedule(dynamic, 500) reduction(+: count)
    for (int node = 0; node < g->num_nodes; ++node) {
        if (distances[node] != step) {
            continue;
        }
        int private_count = 0;
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[node+1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] == NOT_VISITED_MARKER) {
                int index = outgoing / 8;
                int offset = outgoing % 8;
                new_frontier[index] |= (1 << offset);
                //new_frontier->present[outgoing] = true;
                distances[outgoing] = step + 1;
                ++private_count;
            }
        }
        count += private_count;
    }
    return count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(graph* graph, solution* sol) {

    int bitmaplen = graph->num_nodes / 8 + 1;
    unsigned char* bitfrontier = (unsigned char*)malloc(sizeof(char) * bitmaplen);
    unsigned char* bitnewfrontier = (unsigned char*)malloc(sizeof(char) * bitmaplen);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bitmaplen; i++) {
      bitfrontier[i] &= 0;
      bitnewfrontier[i] &= 0;
    }


    // initialize all nodes to NOT_VISITED  
    #pragma omp parallel for schedule(static) 
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    } 
    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    int count = 1;
    while (count != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        count = top_down_step(graph, bitfrontier, bitnewfrontier, sol->distances, step++);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
        
    }
}

void bfs_hybrid(graph* graph, solution* sol) {

    int switchNum = (int)(0.2 * graph->num_nodes);
    int t1 = (int)(0.48 * graph->num_nodes);
    int t2 = (int)(0.2 * graph->num_nodes);

    int bitmaplen = graph->num_nodes / 8 + 1;
    unsigned char* bitfrontier = (unsigned char*)malloc(sizeof(char) * bitmaplen);
    unsigned char* bitnewfrontier = (unsigned char*)malloc(sizeof(char) * bitmaplen);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bitmaplen; i++) {
      bitfrontier[i] &= 0;
      bitnewfrontier[i] &= 0;
    }

    // initialize all nodes to NOT_VISITED 
    #pragma omp parallel for schedule(static) 
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    } 
    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;
    
    int step = 0;
    int count = 1;
    int prev_count = 0;

    while (count != 0) {

        if (count >  switchNum) {
          break;
        }
        
#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif


        count = top_down_step(graph, bitfrontier, bitnewfrontier, sol->distances, step++);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
      } 
      count = 1;
      while (count != 0) {
       
        count = bottom_up_step(graph, bitfrontier, bitnewfrontier,  sol->distances, step++);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
    }
}
