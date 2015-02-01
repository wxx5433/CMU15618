#include "immintrin.h"

void saxpyStream(int N, 
                    float scale, 
                    float X[],
                    float Y[],
                    float result[]) 
{
    for (int i = 0; i < N; i += 8) {
        __m256 vecX = _mm256_load_ps(X + i);
        __m256 vecY = _mm256_load_ps(Y + i);
        __m256 vecScale = _mm256_broadcast_ss(&scale);
        vecX = _mm256_mul_ps(vecScale, vecX);
        _mm256_stream_ps(result + i, _mm256_add_ps(vecX, vecY));
    }
}
