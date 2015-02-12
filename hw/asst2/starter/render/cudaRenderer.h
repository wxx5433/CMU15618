#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

//#define SOLUTION1
//#define SOLUTION2
#define SOLUTION3

class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

#ifdef SOLUTION3
    int* cudaDeviceCirclesInBox;
    int* cudaDeviceBoxCirclesCount;
#endif

#ifdef SOLUTION2
    int* binPixelIndex;
    int* binCircleIndex;
    int* begin;
    int* end;
#endif 

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
};


#endif
