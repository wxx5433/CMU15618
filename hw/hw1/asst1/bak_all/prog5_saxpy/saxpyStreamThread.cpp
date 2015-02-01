#include <pthread.h>

#include "immintrin.h"

typedef struct {
    int N;
    float scale;
    float* X;
    float* Y;
    float* result;
    int threadId;
} WorkerArgs;

void* workerThreadStart(void* threadArgs) {
    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);
    int start = args->threadId * args->N;
    for (int i = start; i < start + args->N; i += 8) {
        __m256 vecX = _mm256_load_ps(args->X + i);
        __m256 vecY = _mm256_load_ps(args->Y + i);
        __m256 vecScale = _mm256_broadcast_ss(&args->scale);
        vecX = _mm256_mul_ps(vecScale, vecX);
        _mm256_stream_ps(args->result + i, _mm256_add_ps(vecX, vecY));
    }
    return NULL;
}

void saxpyStreamThread(int N, 
                    float scale, 
                    float X[],
                    float Y[],
                    float result[]) 
{
    int threadNum = 4;
    int rowsPerThread = N / threadNum;
    pthread_t workers[threadNum];
    WorkerArgs args[threadNum];

    for (int i = 0; i < threadNum; ++i) {
        args[i].N = rowsPerThread;
        args[i].scale = scale;
        args[i].X = X;
        args[i].Y = Y;
        args[i].result = result;
        args[i].threadId = i;
    }

    for (int i = 1; i < threadNum; ++i) {
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);
    }
    workerThreadStart(&args[0]);
    
    // wait for worker threads to complete
    for (int i = 1; i < threadNum; ++i) {
        pthread_join(workers[i], NULL);
    }
}
