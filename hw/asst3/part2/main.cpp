#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>

#include "CycleTimer.h"
#include "graph.h"
#include "bfs.h"

#define USE_BINARY_GRAPH 1

int main(int argc, char** argv) {

    int  num_threads = -1;
    std::string graph_filename;

    if (argc < 2)
    {
        std::cerr << "Usage: <path/to/graph/file> [manual_set_thread_count]\n";
        std::cerr << "To get results across all thread counts: <path/to/graph/file>\n";
        std::cerr << "Run with certain threads count (no correctness run): <path/to/graph/file> <thread_count>\n";
        exit(1);
    }

    int thread_count = -1;
    if (argc == 3)
    {
        thread_count = atoi(argv[2]);
    }

    graph_filename = argv[1];
    graph g;

    printf("----------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    if (thread_count > 0)
    {
        thread_count = std::min(thread_count, omp_get_max_threads());
        printf("Running with %d threads\n", thread_count);
    }
    printf("----------------------------------------------------------\n");

    printf("Loading graph...\n");
    if (USE_BINARY_GRAPH) {
        load_graph_binary(graph_filename.c_str(), &g);
    } else {
        load_graph(argv[1], &g);
        printf("storing binary form of graph!\n");
        store_graph_binary(graph_filename.append(".bin").c_str(), &g);
        exit(1);
    }
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", g.num_edges);
    printf("  Nodes: %d\n", g.num_nodes);
    printf("----------------------------------------------------------\n");

    //If we want to run on all threads
    if (thread_count <= -1)
    {
        //Static assignment to get consistent usage across trials
        int max_threads = omp_get_max_threads();
        int n_usage = (max_threads < 6) ? max_threads : 6;

        int *assignment;

        //static assignments
        int assignment12[6] = {1, 2, 4, 6, 8, 12};
        int assignment8[6] = {1, 2, 3, 4, 6, 8};
        int assignment40[6] = {1, 2, 4, 8, 16, 20};
        int assignment6[6] = {1, 2, 3, 4, 5, 6};

        //dynamic assignment
        if (max_threads == 12)
            assignment = assignment12;
        else if (max_threads == 8)
            assignment = assignment8;
        else if (max_threads == 40)
            assignment = assignment40;
        else if (max_threads == 6)
            assignment = assignment6;
        else {
            if (max_threads < 6)
            {
                int temp[n_usage];
                for (int i = 0; i < n_usage; i++)
                    temp[i] = i+1;
                assignment = temp;
            } else {
                int temp[6];
                temp[0] = 1;
                temp[5] = max_threads;
                for (int i = 1; i < 5; i++)
                    temp[i] = i * max_threads/6;
                assignment = temp;
            }
        }

        solution sol1;
        sol1.distances = (int*)malloc(sizeof(int) * g.num_nodes);
        solution sol2;
        sol2.distances = (int*)malloc(sizeof(int) * g.num_nodes);
        solution sol3;
        sol3.distances = (int*)malloc(sizeof(int) * g.num_nodes);

        double hybrid_base, top_base, bottom_base;
        double hybrid_time, top_time, bottom_time;

        double ref_hybrid_base, ref_top_base, ref_bottom_base;
        double ref_hybrid_time, ref_top_time, ref_bottom_time;

        double start;
        std::stringstream timing;
        std::stringstream ref_timing;

#ifdef USE_HYBRID_FUNCTION
        timing << "Threads  Top Down              Bottom Up             Hybrid\n";
        ref_timing << "Threads  Top Down              Bottom Up             Hybrid\n";

#else
        timing << "Threads  Top Down              Bottom Up\n";
        ref_timing << "Threads  Top Down              Bottom Up\n";

#endif
        //Loop through assignment values;
        for (int i = 0; i < n_usage; i++)
        {
            std::cout << "Running with " << assignment[i] << " threads" << std::endl;
            //Set thread count
            omp_set_num_threads(assignment[i]);

            //Run implementations
            start = CycleTimer::currentSeconds();
            bfs_top_down(&g, &sol1);
            top_time = CycleTimer::currentSeconds() - start;

            //Run reference implementation
            start = CycleTimer::currentSeconds();
            reference_bfs_top_down(&g, &sol1);
            ref_top_time = CycleTimer::currentSeconds() - start;

            //Run implementations
            start = CycleTimer::currentSeconds();
            bfs_bottom_up(&g, &sol2);
            bottom_time = CycleTimer::currentSeconds() - start;

            //Run reference implementation
            start = CycleTimer::currentSeconds();
            reference_bfs_bottom_up(&g, &sol1);
            ref_bottom_time = CycleTimer::currentSeconds() - start;

#ifdef USE_HYBRID_FUNCTION
            start = CycleTimer::currentSeconds();
            bfs_hybrid(&g, &sol3);
            hybrid_time = CycleTimer::currentSeconds() - start;

            //Run reference implementation
            start = CycleTimer::currentSeconds();
            reference_bfs_hybrid(&g, &sol1);
            ref_hybrid_time = CycleTimer::currentSeconds() - start;
#endif

            if (i == 0)
            {
#ifdef USE_HYBRID_FUNCTION
                hybrid_base = hybrid_time;
                ref_hybrid_base = ref_hybrid_time;

#endif
                top_base = top_time;
                bottom_base = bottom_time;
                ref_top_base = ref_top_time;
                ref_bottom_base = ref_bottom_time;

            }

            char buf[1024];
            char ref_buf[1024];

#ifdef USE_HYBRID_FUNCTION
            sprintf(buf, "%4d:   %.4f (%.4fx)     %.4f (%.4fx)     %.4f (%.4fx)\n",
                    assignment[i], top_time, top_base/top_time, bottom_time,
                    bottom_base/bottom_time, hybrid_time, hybrid_base/hybrid_time);
            sprintf(ref_buf, "%4d:   %.4f (%.4fx)     %.4f (%.4fx)     %.4f (%.4fx)\n",
                    assignment[i], ref_top_time, ref_top_base/ref_top_time, ref_bottom_time,
                    ref_bottom_base/ref_bottom_time, ref_hybrid_time, ref_hybrid_base/ref_hybrid_time);

#else
            sprintf(buf, "%4d:   %.4f (%.4fx)     %.4f (%.4fx)\n",
                    assignment[i], top_time, top_base/top_time, bottom_time,
                    bottom_base/bottom_time);

             sprintf(ref_buf, "%4d:   %.4f (%.4fx)     %.4f (%.4fx)\n",
                    assignment[i], ref_top_time, ref_top_base/ref_top_time, ref_bottom_time,
                    ref_bottom_base/ref_bottom_time);

#endif
            timing << buf;
            ref_timing << ref_buf;
        }

        printf("----------------------------------------------------------\n");
        std::cout << "Correctness" << std::endl;

        std::cout << "Testing Correctness of Top Down / Bottom Up\n";
        for (int i=0; i<g.num_nodes; i++) {
            if (sol1.distances[i] != sol2.distances[i]) {
                fprintf(stderr, "*** Results disagree at %d: %d, %d\n", i, sol1.distances[i], sol2.distances[i]);
                exit(1);
            }
        }
#ifdef USE_HYBRID_FUNCTION
        std::cout << "Testing Correctness of Top Down / Hybrid\n";
        for (int i=0; i<g.num_nodes; i++) {
            if (sol1.distances[i] != sol3.distances[i]) {
                fprintf(stderr, "*** Results disagree at %d: %d, %d\n", i, sol1.distances[i], sol3.distances[i]);
                exit(1);
            }
        }
#endif
        std::cout << "Correctness Test Passed\n";
        printf("----------------------------------------------------------\n");
        std::cout << "Timing Summary" << std::endl;
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Reference Summary" << std::endl;
        std::cout << ref_timing.str();
        printf("----------------------------------------------------------\n");
    }
    //Run the code with only one thread count and only report speedup
    else
    {
        solution sol1;
        sol1.distances = (int*)malloc(sizeof(int) * g.num_nodes);

        double hybrid_time, top_time, bottom_time;
        double ref_hybrid_time, ref_top_time, ref_bottom_time;

        double start;
        std::stringstream timing;
        std::stringstream ref_timing;


#ifdef USE_HYBRID_FUNCTION
        timing << "Threads  Top Down    Bottom Up   Hybrid\n";
        ref_timing << "Threads  Top Down    Bottom Up   Hybrid\n";

#else
        timing << "Threads  Top Down    Bottom Up\n";
        ref_timing << "Threads  Top Down    Bottom Up\n";
#endif
        //Loop through assignment values;
        std::cout << "Running with " << thread_count << " threads" << std::endl;
        //Set thread count
        omp_set_num_threads(thread_count);

        //Run implementations
        start = CycleTimer::currentSeconds();
        bfs_top_down(&g, &sol1);
        top_time = CycleTimer::currentSeconds() - start;

        //Run reference implementation
        start = CycleTimer::currentSeconds();
        reference_bfs_top_down(&g, &sol1);
        ref_top_time = CycleTimer::currentSeconds() - start;

        //Run implementations
        start = CycleTimer::currentSeconds();
        bfs_bottom_up(&g, &sol1);
        bottom_time = CycleTimer::currentSeconds() - start;

        //Run reference implementation
        start = CycleTimer::currentSeconds();
        reference_bfs_bottom_up(&g, &sol1);
        ref_bottom_time = CycleTimer::currentSeconds() - start;

#ifdef USE_HYBRID_FUNCTION
        start = CycleTimer::currentSeconds();
        bfs_hybrid(&g, &sol1);
        hybrid_time = CycleTimer::currentSeconds() - start;

        //Run reference implementation
        start = CycleTimer::currentSeconds();
        reference_bfs_hybrid(&g, &sol1);
        ref_hybrid_time = CycleTimer::currentSeconds() - start;
#endif

        char buf[1024];
        char ref_buf[1024];

#ifdef USE_HYBRID_FUNCTION
        sprintf(buf, "%4d:     %.4f     %.4f     %.4f\n",
                thread_count, top_time, bottom_time, hybrid_time);
        sprintf(ref_buf, "%4d:     %.4f     %.4f     %.4f\n",
                thread_count, ref_top_time, ref_bottom_time, ref_hybrid_time);

#else
         sprintf(buf, "%4d:     %.4f     %.4f\n",
                thread_count, top_time, bottom_time);
         sprintf(ref_buf, "%4d:     %.4f     %.4f\n",
                thread_count, ref_top_time, ref_bottom_time);

#endif
        timing << buf;
        ref_timing << ref_buf;
        printf("----------------------------------------------------------\n");
        std::cout << "Timing Summary" << std::endl;
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Reference Summary" << std::endl;
        std::cout << ref_timing.str();
        printf("----------------------------------------------------------\n");
    }

    return 0;
}
