#include <stdlib.h>
#include "immintrin.h"

// fabs(guess * guess * x - 1.f)
inline __m256 fabs(__m256 guess, __m256 vecx,
        __m256 floatOne, __m256 floatZero) {
    __m256 error = _mm256_sub_ps(_mm256_mul_ps(
                _mm256_mul_ps(guess, guess), vecx), floatOne);
    // get absolute value
    __m256 minusError = _mm256_sub_ps(floatZero, error);
    __m256 absError = _mm256_max_ps(error, minusError);
    return absError;
}

void sqrtIntrinsics(int N,
                    float initialGuess,
                    float values[],
                    float output[])
{
    static const float kThreshold = 0.00001f;
    static const int VECTOR_WIDTH = 8;

    // some constant vec to reuse
    __m256 floatOne = _mm256_set1_ps(1.f);
    __m256 floatPointFive = _mm256_set1_ps(0.5f);
    __m256 floatZero = _mm256_set1_ps(0.f);
    __m256 floatThree = _mm256_set1_ps(3.f);
    __m256 vecThreshold = _mm256_set1_ps(kThreshold);
    __m256 vecAllOne = _mm256_set1_ps(-1.f);

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // x = values[i]
        __m256 vecx = _mm256_load_ps(values + i);       
        // guess = initialGuess
        __m256 guess = _mm256_set1_ps(initialGuess);
        // error = fabs(guess * guess * x - 1.f)
        __m256 error = fabs(guess, vecx, floatOne, floatZero);
        // test if error - kThreshold > 0
        __m256 cmpRes = _mm256_cmp_ps(error, vecThreshold, _CMP_GT_OQ);
        int res = _mm256_testz_ps(cmpRes, vecAllOne);
        while (res == 0) {
            // tmp1 = 3.f * guess
            __m256 tmp1 = _mm256_mul_ps(floatThree, guess);
            // tmp2 = x * guess * guess * guess
            __m256 tmp2 = _mm256_mul_ps(_mm256_mul_ps(
                        _mm256_mul_ps(vecx, guess), guess), guess);
            // guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            guess = _mm256_sub_ps(tmp1, tmp2);
            guess = _mm256_mul_ps(guess, floatPointFive);
            // get error
            error = fabs(guess, vecx, floatOne, floatZero);
            cmpRes = _mm256_cmp_ps(error, vecThreshold, _CMP_GT_OQ);
            res = _mm256_testz_ps(cmpRes, vecAllOne);
        }
        __m256 result = _mm256_mul_ps(vecx, guess);
        _mm256_store_ps(output + i, result);
    }
}

