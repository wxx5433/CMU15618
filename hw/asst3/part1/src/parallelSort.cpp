/* Copyright 2014 15418 Staff */

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>

#include "parallelSort.h"
#include "CycleTimer.h"

using namespace std;

#define NO_DEBUG

void printArr(const char* arrName, int *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %d %d %d %d\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void printArr(const char* arrName, float *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %f %f %f %f\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void randomSample(float *data, size_t dataSize, float *sample, size_t sampleSize) {
  int i;
  
  srand(time(NULL));

  for (i=0; i< sampleSize; i++) {
    sample[i] = data[i];    
  }
  for (; i < dataSize; ++i) {
    int j = rand() % (i + 1);
    if (j < sampleSize) {
      sample[j] = data[i];
    }
  }
}

int compare( const void* n1, const void *n2) {
  return (*(int*)n1 - *(int*)n2);

}

typedef struct
{
  float* array;
  int index;
} Bucket;


void parallelSort(float *data, float *&sortedData, int procs, int procId, size_t dataSize, size_t &localSize) {
  // Implement parallel sort algorithm as described in assignment 3
  // handout. 
  // Input:
  //  data[]: input arrays of unsorted data, *distributed* across p processors
  //          note that data[] array on each process contains *a fraction* of all data
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  //  procs: total number of processes
  //  procId: id of this process
  //  dataSize: aggregated size of *all* data[] arrays
  //  localSize: size of data[] array on *this* process (as input)
  //             size of sortedData[] array on *this* process (as output)
  //
  //
  // Step 1: Choosing Pivots to Define Buckets
  // Step 2: Bucketing Elements of the Input Array
  // Step 3: Redistributing Elements
  // Step 5: Final Local Sort
  // ***********************************************************************
  
  double start, end;  
  // Step 1
  int s = 12 * log(dataSize);
  
  float *pivots = (float*)malloc(sizeof(float) * (procs - 1));

  float *pivotGlobal;
  start = CycleTimer::currentSeconds();
  if (procId == 0) {
    pivotGlobal = (float*)malloc(sizeof(float) * s * procs);
    
    randomSample(data, localSize, pivotGlobal, s);
    
    MPI_Gather(pivotGlobal, s, MPI_FLOAT, pivotGlobal, s, MPI_FLOAT, 0, MPI_COMM_WORLD);
    sort(pivotGlobal, pivotGlobal + s * procs);

    pivots[0] = 0;
    for (int k = 1; k < procs; k++) {
      pivots[k - 1] = pivotGlobal[k * s];
    }

    MPI_Bcast(pivots, procs - 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    
  } else {
      float *pivotProc = (float*)malloc(sizeof(float) * s);
      
      randomSample(data, localSize, pivotProc, s);
      //MPI_Send(pivotProc, s, MPI_FLOAT, 0, procId, MPI_COMM_WORLD);
      MPI_Gather(pivotProc, s, MPI_FLOAT, pivotGlobal, s, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(pivots, procs - 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  end = CycleTimer::currentSeconds();
  printf("sampling takes %f ms\n", (end - start) * 1000);
  start = CycleTimer::currentSeconds();
  
  // Step 2

 
  vector<vector<float> > buckets(procs);
 
  int* counts = (int*)malloc(sizeof(int) * procs);
  int* disp   = (int*)malloc(sizeof(int) * procs);
 
  memset(counts, 0, sizeof(int) * procs);

  start = CycleTimer::currentSeconds();
  for (int i = 0; i < localSize; i++) {
    int bid = lower_bound(pivots, pivots + procs - 1, data[i]) - pivots;
   
    buckets[bid].push_back(data[i]);
    counts[bid]++;
  }
  end = CycleTimer::currentSeconds();
  printf("Finding buckets took %f\n", end - start); 
  
  int *recCounts = (int*)malloc(sizeof(int) * procs);
  //int *counts = (int*)malloc(sizeof(int) * procs);
  
  int *displacement = (int*)malloc(sizeof(int) * procs);
  
  int d = 0;
  for (int x = 0; x < procs; x++) {
    //counts[x] = buckets[x].size();
    displacement[x] = d;
    d += counts[x];
  }
  

  MPI_Alltoall(counts, 1, MPI_INT, recCounts, 1, MPI_INT, MPI_COMM_WORLD);
  int tmp = 0;

  for (int i = 0; i < procs; ++i) {
    disp[i] = tmp;
    tmp += recCounts[i];
  }

  float* bucketsArray = (float*)malloc(sizeof(float) * localSize);
  int index = 0;
  for (int i = 0; i < procs; ++i) {
    for (int j = 0; j < counts[i]; ++j) {
      bucketsArray[index] = buckets[i][j];
      index++;
    }
  }
  end = CycleTimer::currentSeconds();
  printf("step 2 takes %f ms\n", (end - start) * 1000);
  
  start = CycleTimer::currentSeconds();
  
  //////////////////////////// Step 3 ////////////////////////////////////////////
  localSize = 0;
  for (int i = 0; i < procs; i++) {
    localSize += recCounts[i];
  }
  sortedData = (float*)malloc(sizeof(float) * localSize);
  
  MPI_Alltoallv(bucketsArray, counts, displacement, MPI_FLOAT, sortedData, recCounts, disp, MPI_FLOAT, MPI_COMM_WORLD);
   
  
  end = CycleTimer::currentSeconds();
  printf("process %d exchanging data takes %f ms\n", procId, (end - start) * 1000);
  

  //////////////////////////////////////// Step 4 ///////////////////////////////////////////////
  
  start = CycleTimer::currentSeconds();
  sort (sortedData, sortedData + localSize);
  end  = CycleTimer::currentSeconds(); 
  printf("Sorting takes %f\n", (end - start) * 1000);  

}

