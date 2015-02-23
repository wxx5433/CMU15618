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

int bottom_up_step(graph* g, int* distances, int step) {
#ifdef DEBUG
    printf("bottom up\n");
#endif
    int count = 0;
    #pragma omp parallel for schedule(dynamic, 400) reduction(+:count)
    for (int node = 0; node < g->num_nodes; node++) {
        // already find distances
        if (distances[node] != NOT_VISITED_MARKER) {
            continue;
        }
        int private_count = 0;
        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges: g->outgoing_starts[node + 1];
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            // nodes just find in last step
            if (distances[incoming] == step) {
                distances[node] = step + 1;
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
    return count;
}

void bfs_bottom_up(graph* graph, solution* sol)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    int count = 1;
    while (count != 0) {
#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        count = bottom_up_step(graph, sol->distances, step++);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif
    }
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
int top_down_step(graph* g, int* distances, int step) {
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

    //vertex_set list1;
    //vertex_set list2;
    //vertex_set_init(&list1, graph->num_nodes);
    //vertex_set_init(&list2, graph->num_nodes);

    //vertex_set* frontier = &list1;
    //vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(static)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    //frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    int count = 1;
    while (count != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        //vertex_set_clear(new_frontier);

        count = top_down_step(graph, sol->distances, step++);
        //top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("%.4f sec\n", end_time - start_time);
#endif

        // swap pointers
        //vertex_set* tmp = frontier;
        //frontier = new_frontier;
        //new_frontier = tmp;
    }
}

void bfs_hybrid(graph* graph, solution* sol) {
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(static)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    int switchNum = (int)(0.07 * graph->num_nodes);
#ifdef DEBUG
    printf("switch num: %d\n", switchNum);
#endif
    int step = 0;
    int count = 1;
    while (count != 0) {
        if (count > switchNum) {
            break;
        }
        count = top_down_step(graph, sol->distances, step++);
    }
    count = 1;
    while (count != 0) {
        count = bottom_up_step(graph, sol->distances, step++);
    }
}
