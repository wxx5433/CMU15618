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
    list->present = (bool*)malloc(sizeof(bool) * list->alloc_count);
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
            if (frontier[index] & (1 << offset)) { //frontier->present[incoming] == true) {//distances[incoming] == step) {
                //printf("doing node %d with incoming node %d in step %d \n", node, incoming, step);
                distances[node] = step + 1;
                //frontier->present[incoming] = false;
                //new_frontier->present[node] = true;
                index = node / 8;
                offset = node % 8;
                new_frontier[index] |= (1 << offset);
                ++private_count;
                //if (__sync_bool_compare_and_swap(&distances[node], NOT_VISITED_MARKER, step + 1)) {
                    //int index = __sync_fetch_and_add(&new_frontier->count, 1);
                    //new_frontier->present[index] = node;
                //}
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
    //int* pdistance = (int*)malloc(sizeof(int) * graph->num_nodes * 4); // padded distance
    vertex_set list1;
    vertex_set list2;

    vertex_set_init(&list2, graph->num_nodes);
    vertex_set_init(&list1, graph->num_nodes);
  
    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
   
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
        //frontier->present[i] = false;
        //new_frontier->present[i] = false;
    }
/*
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < graph->num_nodes; i++) {
       pdistance[i * 4] = NOT_VISITED_MARKER;

    }*/
    bitfrontier[ROOT_NODE_ID] |= 1;

    //printf("here printing %x \n", bitfrontier[ROOT_NODE_ID]);
    //frontier->present[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;
    //pdistance[ROOT_NODE_ID] = 0;

    int step = 0;
    int count = 1;
    while (count != 0) {
#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        //count = bottom_up_step(graph, frontier, new_frontier, sol->distances, step++);
        count = bottom_up_step(graph, bitfrontier, bitnewfrontier, sol->distances, step++);

        //vertex_set* tmp = frontier;
        //frontier = new_frontier;
        //new_frontier = tmp;
       // double start = CycleTimer::currentSeconds();
       // #pragma omp parallel for schedule(static)
       // for (int i = 0; i < graph->num_nodes; i++) {
         // new_frontier->present[i] = false;
       // }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
        //double end = CycleTimer::currentSeconds();
        //printf("%f\n", end - start);
        /*
        #pragma parallel for schedule(dynamic, 400)
        for (int i = 1; i < graph->num_nodes; ++i) {
          if (frontier->present[i]) {
            sol->distances[i] = step + 1;
            frontier->present[i] = 0;
          }
        }
        step++;*/
        //count = bottom_up_step(graph, pdistance, step++);
#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif
    }
    /*
    sol->distances = (int*)malloc(sizeof(int) * graph->num_nodes); 
    
   int tmp = 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < graph->num_nodes; i++) {
      sol->distances[i]  = pdistance[i * 4];
    }
    free(pdistance);
    */
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
            /*
            if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
                int index = __sync_fetch_and_add(&new_frontier->count, 1);
                new_frontier->present[index] = outgoing
            }
            */
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

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    
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
        frontier->present[i] = false;
        new_frontier->present[i] = false;
    } 
    // setup frontier with the root node
    frontier->present[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    int count = 1;
    while (count != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        //vertex_set_clear(new_frontier);
        count = top_down_step(graph, bitfrontier, bitnewfrontier, sol->distances, step++);
        //count = top_down_step(graph, frontier, new_frontier, sol->distances, step++);
        //top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
        
        /*
        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < graph->num_nodes; i++) {
          new_frontier->present[i] = false;

        }*/

        
    }
}

void bfs_hybrid(graph* graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

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
        frontier->present[i] = false;
        new_frontier->present[i] = false;
    } 
    // setup frontier with the root node
    frontier->present[ROOT_NODE_ID] = true;
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

        //vertex_set_clear(new_frontier);

        count = top_down_step(graph, bitfrontier, bitnewfrontier, sol->distances, step++);
        //top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
        /*
        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < graph->num_nodes; i++) {
          new_frontier->present[i] = false;

        }*/
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
   /*
    count = 1;
   
     while (count != 0) {
        
        count = bottom_up_step(graph, bitfrontier, bitnewfrontier,  sol->distances, step++);
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < graph->num_nodes; i++) {
          new_frontier->present[i] = false;

        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bitmaplen; i++) {
          bitfrontier[i] = bitnewfrontier[i];
          bitnewfrontier[i] = 0;
        }
    }*/
    
/*
    vertex_set list3;
    vertex_set list4;
    vertex_set_init(&list3, graph->num_nodes);
    vertex_set_init(&list4, graph->num_nodes);

    vertex_set* frontier = &list3;
    vertex_set* new_frontier = &list4;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(static)
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier->present[i] = false;
        new_frontier->present[i] = false;
    }

    // setup frontier with the root node
    frontier->present[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;
  
//    int switchNum = (int)(0.07 * graph->num_nodes);

    int step = 0;
    int count = 1;
    while (count != 0) {
 //       if (count > switchNum) 
  //        break;
#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        //vertex_set_clear(new_frontier);

        count = top_down_step(graph, frontier, new_frontier, sol->distances, step++);
        //top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < graph->num_nodes; i++) {
          new_frontier->present[i] = false;

        }
*/
    //}


    /*
    count = 1;
   
    while (count != 0) {
        
        count = bottom_up_step(graph, frontier, new_frontier,  sol->distance
 */
}
