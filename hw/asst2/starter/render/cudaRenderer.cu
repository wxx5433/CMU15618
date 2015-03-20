#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>  // use device_ptr

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "cycleTimer.h"

#include "circleBoxTest.cu_inl"

//#define DEBUG

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// some contants
#define BOX_SIZE 16

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles(int*& pixelIndex, int*& circleIndex) { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

/************** my bin solution **************/
// myShadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
myShadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

#ifdef SOLUTION2
/************* solution 2: render by circles and sort ****************/
__global__ void kernelInitValue(int* binPixelIndex, int initVal, int size) { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    binPixelIndex[index] = initVal;
}

__global__ void myBinKernelRenderCircles(int* binCircleIndex, int* begin, int* end, int pixelNum) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height) {
        return;
    }

    int invWidth = 1.f / width;
    int invHeight = 1.f / height;
    
    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(imageX) + 0.5f),
        invHeight * (static_cast<float>(imageY) + 0.5f));
    int pixelIndex = imageY * width + imageX;
    int offset = 4 * pixelIndex;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[offset]);

    int b = begin[pixelIndex];
    int e = end[pixelIndex];

    if (b == e) {
        return;
    }

    //printf("%d\t%d\n", b, e);

    for (int i = b; i < e; ++i) {
        int index = binCircleIndex[i];
//        printf("%d\n", index);
        int index3 = 3 * index;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        myShadePixel(index, pixelCenterNorm, p, imgPtr);
    }
}

__global__ void kernelFindBeginAndEnd(int* binPixelIndex, int* begin, int* end, int totalSize, int pixelNum) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalSize || binPixelIndex[index] == pixelNum) {
        return;
    }
    int pixelIndex = binPixelIndex[index];
    if (index == 0 || pixelIndex != binPixelIndex[index - 1]) {
        begin[pixelIndex] = index;
#ifdef DEBUG
        if (index % 768 == 0) {
            printf("%d\t%d\n", index, pixelIndex);
        }
#endif
    }
    if (index == totalSize - 1 || pixelIndex != binPixelIndex[index + 1]) { 
        end[pixelIndex] = index + 1;
    }
}

__global__ void kernelFindBin(int* binPixelIndex, int* binCircleIndex) { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int binIndex = index * imageWidth * imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            float diffX = p.x - pixelCenterNorm.x;
            float diffY = p.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            float rad = cuConstRendererParams.radius[index];
            float maxDist = rad * rad;
            if (pixelDist <= maxDist) {
                binCircleIndex[binIndex] = index;
                int pixelIndex = pixelY * imageWidth + pixelX;
                binPixelIndex[binIndex++] = pixelIndex;
            }
            imgPtr++;
        }
    }
}
/********************* end of solution 2 ***************************/
#endif

#ifdef SOLUTION1
/************** Soluton 1: render by cells ********************/
// myKernelRenderCells-- (CUDA device code)
//
// Each thread renders a cell. Loop through all circles to decide 
// if it makes a contribution to the cell. 
// Note: This method may be inefficient when the number of cells
//       far less than the number of circles.  
__global__ void myKernelRenderCells() {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    // check boundary
    if (imageX >= width || imageY >= height) {
        return;
    }

    float invWidth = 1.f / width;
    float invHeight = 1.f / height;

    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(imageX) + 0.5f),
        invHeight * (static_cast<float>(imageY) + 0.5f));
    int offset = 4 * (imageY * width + imageX);
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[offset]);
    
    // loop through all circles
    int circleNum = cuConstRendererParams.numCircles;
    for (int circleIndex = 0; circleIndex < circleNum; ++circleIndex) {
        int index3 = circleIndex * 3;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
    }
}
/**************** End of code solution 1: render by cells ******************/
#endif

#ifdef SOLUTION3
/************** solution 3: render by blocks(bloom filter) ****************/
__global__ void getCirclesInBox(int* cudaDeviceCirclesInBox, int* cudaDeviceBoxCirclesCount, 
    int boxNum_x, int boxNum_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= boxNum_x || y >= boxNum_y) {
        return;
    }

    short width = cuConstRendererParams.imageWidth;
    short height = cuConstRendererParams.imageHeight;

    float invWidth = 1.f / width;
    float invHeight = 1.f / height;


    // get the boundary of the box
    float boxL = invWidth * static_cast<float>(x * BOX_SIZE);
    float boxR = invWidth * static_cast<float>((x + 1) * BOX_SIZE);
    float boxT = invHeight * static_cast<float>((y + 1) * BOX_SIZE);
    float boxB = invHeight * static_cast<float>(y * BOX_SIZE);

    int circlesNum = cuConstRendererParams.numCircles;
    int boxIndex = boxNum_x * y + x;
    int boxOffset = circlesNum * boxIndex;
    int circlesCount = 0;
    // walk through all circles to find those intersect with the box
    for (int index = 0; index < circlesNum; ++index) {
        int index3 = index * 3;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[index];

        if (circleInBox(p.x, p.y, rad, boxL, boxR, boxT, boxB)) {
            cudaDeviceCirclesInBox[boxOffset + circlesCount] = index;
            ++circlesCount;
        }
    }

    //printf("circle count %d\n", circlesCount);
    cudaDeviceBoxCirclesCount[boxIndex] = circlesCount;
}

__global__ void kernelRenderCirclesByBox(int* cudaDeviceCirclesInBox, int* cudaDeviceBoxCirclesCount, 
    int boxNum_x, int boxNum_y) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    short width = cuConstRendererParams.imageWidth;
    short height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height) {
        return;
    }

    float invWidth = 1.f / width;
    float invHeight = 1.f / height;

    float2 pixelCenterNorm = make_float2(
        invWidth * (static_cast<float>(imageX) + 0.5f),
        invHeight * (static_cast<float>(imageY) + 0.5f));
    int offset = 4 * (imageY * width + imageX);
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[offset]);
    
    // loop through all circles in the box
    int index_x = imageX / BOX_SIZE; 
    int index_y = imageY / BOX_SIZE;
    int boxIndex = index_y * boxNum_x + index_x;
    int circlesInBoxNum = cudaDeviceBoxCirclesCount[boxIndex];
    int numCircles = cuConstRendererParams.numCircles;
    for (int i = 0; i < circlesInBoxNum; ++i) {
        int index = cudaDeviceCirclesInBox[numCircles * boxIndex + i]; 
        int index3 = index * 3;
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        shadePixel(index, pixelCenterNorm, p, imgPtr);
    }
}
/********************* End of solution 3: render by box ******************/
#endif

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

#ifdef SOLUTION3
    cudaDeviceCirclesInBox = NULL;
    cudaDeviceBoxCirclesCount = NULL;
#endif

#ifdef SOLUTION2
    binPixelIndex = NULL;
    binCircleIndex = NULL;
    begin = NULL;
    end = NULL;
#endif
    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image->data;   // delete data first!!
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        printf("free space!\n");
#ifdef SOLUTION3
        cudaFree(cudaDeviceCirclesInBox);
        cudaFree(cudaDeviceBoxCirclesCount);
#endif

#ifdef SOLUTION2
        cudaFree(binPixelIndex);
        cudaFree(binCircleIndex);
        cudaFree(begin);
        cudaFree(end);
#endif
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
    cudaFree(cuConstNoiseYPermutationTable);
    cudaFree(cuConstNoiseXPermutationTable);
    cudaFree(cuConstNoise1DValueTable);
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy


#ifdef SOLUTION3
    short boxNum_x = (image->width + BOX_SIZE - 1) / BOX_SIZE;
    short boxNum_y = (image->height + BOX_SIZE - 1) / BOX_SIZE;
    cudaMalloc(&cudaDeviceCirclesInBox, sizeof(int) * numCircles * boxNum_x * boxNum_y);
    cudaMalloc(&cudaDeviceBoxCirclesCount, sizeof(int) * boxNum_x * boxNum_y);
#endif

#ifdef SOLUTION2
    int totalPixelNum = image->width * image->height;
    int totalSize = numCircles * totalPixelNum;
    cudaMalloc(&binPixelIndex, sizeof(int) * totalSize);
    cudaMalloc(&binCircleIndex, sizeof(int) * totalSize);
    cudaMalloc(&begin, sizeof(int) * totalPixelNum);
    cudaMalloc(&end, sizeof(int) * totalPixelNum);
    // initialize to max, so that after sorting it appears at last
    dim3 blockDim1(256, 1);
    dim3 gridDim1((totalSize + blockDim1.x - 1) / blockDim1.x);
    kernelInitValue<<<gridDim1, blockDim1>>>(binPixelIndex, totalPixelNum, totalSize);
    // find the start and end index of each bin
    dim3 blockDim2(256, 1);
    dim3 gridDim2((totalPixelNum + blockDim2.x - 1) / blockDim2.x);
    kernelInitValue<<<gridDim2, blockDim2>>>(begin, -1, totalPixelNum);
    kernelInitValue<<<gridDim2, blockDim2>>>(end, -1, totalPixelNum);
#endif

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image) {
        delete image->data;   // delete data as well!!!!
        delete image;
    }
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaThreadSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaThreadSynchronize();
}

void
CudaRenderer::render() {
#ifdef SOLUTION1
    // solution 1: create threads acoording to cells, get 21/65
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x, 
        (image->height + blockDim.y - 1) / blockDim.y);

    myKernelRenderCells<<<gridDim, blockDim>>>();
    cudaThreadSynchronize();
#endif

#ifdef SOLUTION2
    // solution 2: data-parallel approach, get 2/65, out of memory
    int totalPixelNum = image->width * image->height;
    int totalSize = numCircles * totalPixelNum;

    // put all pixel into circle bins
    double startTime = CycleTimer::currentSeconds();
    dim3 blockDim1(256, 1);
    dim3 gridDim1((numCircles + blockDim1.x - 1) / blockDim1.x);
    kernelFindBin<<<gridDim1, blockDim1>>>(binPixelIndex, binCircleIndex);   
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    printf("time to put into bin: %.4f ms\n", 1000 * (endTime - startTime));

    // sort bins by pixel index
    // must wrap device memory with device_ptr
    startTime = CycleTimer::currentSeconds();
    thrust::device_ptr<int> dev_binPixelIndex(binPixelIndex);
    thrust::device_ptr<int> dev_binCircleIndex(binCircleIndex);
    thrust::stable_sort_by_key(dev_binPixelIndex, dev_binPixelIndex + totalSize, dev_binCircleIndex);
    endTime = CycleTimer::currentSeconds();
    printf("time to sort: %.4f ms\n", 1000 * (endTime - startTime));

    startTime = CycleTimer::currentSeconds();
    dim3 blockDim2(256, 1);
    dim3 gridDim2((totalSize + blockDim2.x - 1) / blockDim2.x);
    kernelFindBeginAndEnd<<<gridDim2, blockDim2>>>(binPixelIndex, 
        begin, end, totalSize, totalPixelNum);
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("time to find begin and end: %.4f ms\n", 1000 * (endTime - startTime));

    // render every pixel
    startTime = CycleTimer::currentSeconds();
    dim3 blockDim3(16, 16, 1);
    dim3 gridDim3((image->width + blockDim3.x - 1) / blockDim3.x,
                    (image->height + blockDim3.y - 1) / blockDim3.y);
    myBinKernelRenderCircles<<<gridDim3, blockDim3>>>(binCircleIndex, begin, end, totalPixelNum);
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("time to render by pixel: %.4f ms\n", 1000 * (endTime - startTime));
#endif 

    // solution 3: bloom filter, get 65/65
#ifdef SOLUTION3
    short boxNum_x = (image->width + BOX_SIZE - 1) / BOX_SIZE;
    short boxNum_y = (image->height + BOX_SIZE - 1) / BOX_SIZE;
    dim3 blockDim(8, 8);
    dim3 gridDim((boxNum_x + blockDim.x - 1) / blockDim.x, 
                (boxNum_y + blockDim.y - 1) / blockDim.y);

    getCirclesInBox<<<blockDim, gridDim>>>(cudaDeviceCirclesInBox, cudaDeviceBoxCirclesCount, boxNum_x, boxNum_y);
    cudaDeviceSynchronize();
    dim3 blockDim2(16, 16, 1);
    dim3 gridDim2(
        (image->width + blockDim2.x - 1) / blockDim2.x, 
        (image->height + blockDim2.y - 1) / blockDim2.y);

    kernelRenderCirclesByBox<<<gridDim2, blockDim2>>>(cudaDeviceCirclesInBox, cudaDeviceBoxCirclesCount, boxNum_x, boxNum_y);
    cudaDeviceSynchronize();
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        printf("cada failed: %s\n", cudaGetErrorString(cudaerr));
    }
#endif
}
