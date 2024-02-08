#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <npp.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 1
#define IN_TILE_DIM 4
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

float *randomMatrix(int width, int height) {
    float *matrix = (float *) malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        matrix[i] = (float) rand() / RAND_MAX;
    }
    return matrix;
}

float *onesMatrix(int width, int height) {
    float *matrix = (float *) malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        matrix[i] = 1.0f;
    }
    return matrix;
}

__constant__ float kFilter_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void conv2DTiledConstKernel(float *input, float *output,
                                       int width, int height) {
    __shared__ float inputTile[IN_TILE_DIM][IN_TILE_DIM];
    // Input tile coordinates
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    if (row < height && col < width) {
        inputTile[threadIdx.y][threadIdx.x] = input[row * width + col];
    } else {
        inputTile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Output tile coordinates
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // In a valid convolution, the output is smaller than the input
    row -= FILTER_RADIUS;
    col -= FILTER_RADIUS;

    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
        float sum = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                sum += inputTile[tileRow + fRow][tileCol + fCol] * kFilter_d[fRow][fCol];
            }
        }
        output[row * (width - 2 * FILTER_RADIUS) + col] = sum;
    }
}

__global__ void conv2DTiledCachedConstKernel(float *input, float *output,
                                             int width, int height) {
    __shared__ float inputTile[IN_TILE_DIM][IN_TILE_DIM];
    // Input tile coordinates
    int col = blockIdx.x * IN_TILE_DIM + threadIdx.x;
    int row = blockIdx.y * IN_TILE_DIM + threadIdx.y;
    if (row < height && col < width) {
        inputTile[threadIdx.y][threadIdx.x] = input[row * width + col];
    } else {
        inputTile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (row < FILTER_RADIUS || col < FILTER_RADIUS || col >= (width - FILTER_RADIUS) || row >= (height - FILTER_RADIUS)) return;

    // Output tile coordinates
    row -= FILTER_RADIUS;
    col -= FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    float sum = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
            // If this value is in shared memory, access it there
            if (tileCol + fCol >= 0 &&
                tileCol + fCol < IN_TILE_DIM &&
                tileRow + fRow >= 0 &&
                tileRow + fRow < IN_TILE_DIM) {
                sum += inputTile[tileRow + fRow][tileCol + fCol] * kFilter_d[fRow][fCol];
            } else {
                // Otherwise, access it from global memory
                sum += input[(row + fRow) * width + (col + fCol)] * kFilter_d[fRow][fCol];
            }
        }
    }

    output[row * (width - 2 * FILTER_RADIUS) + col] = sum;
}

void simpleConvolution(float* pSrc, float* pDst, Npp32f* pKernel, NppiSize oSrcSize, NppiSize oKernelSize) {
    // NppiSize oSrcROI = {oSrcSize.width - oKernelSize.width + 1, oSrcSize.height - oKernelSize.height + 1};
    NppiSize oSrcROI = {oSrcSize.width, oSrcSize.height};
    NppiPoint oAnchor = {oKernelSize.width / 2, oKernelSize.height / 2};

    // Allocate memory on device
    float *pSrcDev, *pDstDev, *pKernelDev;
    cudaMalloc(&pSrcDev, oSrcSize.width * oSrcSize.height * sizeof(float));
    cudaMalloc(&pDstDev, oSrcROI.width * oSrcROI.height * sizeof(float));
    cudaMalloc(&pKernelDev, oKernelSize.width * oKernelSize.height * sizeof(float));

    // Copy data to device
    cudaMemcpy(pSrcDev, pSrc, oSrcSize.width * oSrcSize.height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pKernelDev, pKernel, oKernelSize.width * oKernelSize.height * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    NppStatus status = nppiFilter_32f_C1R(
        (const Npp32f *)pSrcDev, oSrcSize.width * sizeof(float),
        (Npp32f *)pDstDev, oSrcROI.width * sizeof(float), oSrcROI, pKernelDev,
        oKernelSize, {1, 1});

    // Check status
    if (status != NPP_SUCCESS) {
        printf("NPP Error: %d\n", status);
        // Handle error...
    }

    // Copy result back to host
    cudaMemcpy(pDst, pDstDev, oSrcROI.width * oSrcROI.height * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(pSrcDev);
    cudaFree(pDstDev);
    cudaFree(pKernelDev);
}

void testTiledConvolution(float *input, int width, int height) {
    // Allocate memory on device
    float *inputDev, *outputDev;
    cudaMalloc(&inputDev, width * height * sizeof(float));
    cudaMalloc(&outputDev, (width - 2 * FILTER_RADIUS) * (height - 2 * FILTER_RADIUS) * sizeof(float));

    // Copy data to device
    cudaMemcpy(inputDev, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    dim3 gridSize(width / OUT_TILE_DIM - 1, height / OUT_TILE_DIM - 1, 1);
    dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM, 1);

    conv2DTiledConstKernel<<<gridSize, blockSize>>>(inputDev, outputDev, width, height);

    // Copy result back to host
    float *output = (float *) malloc((width - 2 * FILTER_RADIUS) * (height - 2 * FILTER_RADIUS) * sizeof(float));
    cudaMemcpy(output, outputDev, (width - 2 * FILTER_RADIUS) * (height - 2 * FILTER_RADIUS) * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(inputDev);
    cudaFree(outputDev);

    printf("Tiled Convolution\n");

    // Print the output matrix as a nice grid
    for (int row = 0; row < height - 2 * FILTER_RADIUS; row++) {
        for (int col = 0; col < width - 2 * FILTER_RADIUS; col++) {
            printf("%.2f ", output[row * (width - 2 * FILTER_RADIUS) + col]);
        }
        printf("\n");
    }

    free(output);
}

void testCachedConvolution(float *input, int width, int height) {
    // Allocate memory on device
    float *inputDev, *outputDev;
    cudaMalloc(&inputDev, width * height * sizeof(float));
    cudaMalloc(&outputDev, (width - 2 * FILTER_RADIUS) * (height - 2 * FILTER_RADIUS) * sizeof(float));

    // Copy data to device
    cudaMemcpy(inputDev, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    dim3 gridSize(width / IN_TILE_DIM, height / IN_TILE_DIM, 1);
    dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM, 1);

    conv2DTiledCachedConstKernel<<<gridSize, blockSize>>>(inputDev, outputDev, width, height);

    // Copy result back to host
    float *output = (float *) malloc((width - 2 * FILTER_RADIUS) * (height - 2 * FILTER_RADIUS) * sizeof(float));
    cudaMemcpy(output, outputDev, (width - 2 * FILTER_RADIUS) * (height - 2 * FILTER_RADIUS) * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(inputDev);
    cudaFree(outputDev);

    printf("Cached Convolution\n");

    // Print the output matrix as a nice grid
    for (int row = 0; row < height - 2 * FILTER_RADIUS; row++) {
        for (int col = 0; col < width - 2 * FILTER_RADIUS; col++) {
            printf("%.2f ", output[row * (width - 2 * FILTER_RADIUS) + col]);
        }
        printf("\n");
    }

    free(output);
}


int main(int argc, char **argv) {
    const size_t width = 4;
    const size_t height = 4;

    // Allocate memory for the input and output matrices
    float *inputMatrix_h = randomMatrix(width, height);

    // Create Gaussian blur kernel
    float filter[FILTER_RADIUS * 2 + 1][FILTER_RADIUS * 2 + 1];
    float filterSum = 0.0f;
    for (int row = 0; row < FILTER_RADIUS * 2 + 1; row++) {
        for (int col = 0; col < FILTER_RADIUS * 2 + 1; col++) {
            float x = col - FILTER_RADIUS;
            float y = row - FILTER_RADIUS;
            filter[row][col] = expf(-(x * x + y * y) / (2 * 1.0f)) / (2 * M_PI * 1.0f);
            filterSum += filter[row][col];
        }
    }
    for (int row = 0; row < FILTER_RADIUS * 2 + 1; row++) {
        for (int col = 0; col < FILTER_RADIUS * 2 + 1; col++) {
            filter[row][col] /= filterSum;
        }
    }

    // Create an identity kernel
    // float filter[FILTER_RADIUS * 2 + 1][FILTER_RADIUS * 2 + 1];
    // for (int row = 0; row < FILTER_RADIUS * 2 + 1; row++) {
    //     for (int col = 0; col < FILTER_RADIUS * 2 + 1; col++) {
    //         filter[row][col] = 0.0f;
    //     }
    // }

    // filter[FILTER_RADIUS][FILTER_RADIUS] = 1.f;
    // filter[0][0] = 1.f;
    // filter[2][2] = 1.f;

    // Create an Npp32s version of the kernel
    Npp32f *filter32s = (Npp32f *)malloc((2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(Npp32f));
    for (int row = 0; row < FILTER_RADIUS * 2 + 1; row++) {
        for (int col = 0; col < FILTER_RADIUS * 2 + 1; col++) {
            filter32s[row * (2 * FILTER_RADIUS + 1) + col] = filter[row][col];
        }
    }

    // Print the input matrix as a nice grid
    printf("Input Matrix\n");
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%.2f ", inputMatrix_h[row * width + col]);
        }
        printf("\n");
    }

    // Print the filter as a nice grid
    printf("Filter\n");
    for (int row = 0; row < FILTER_RADIUS * 2 + 1; row++) {
        for (int col = 0; col < FILTER_RADIUS * 2 + 1; col++) {
            printf("%.2f ", filter[row][col]);
        }
        printf("\n");
    }

    // Copy the filter to the constant memory
    cudaMemcpyToSymbol(kFilter_d, filter, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));

    testTiledConvolution(inputMatrix_h, width, height);
    testCachedConvolution(inputMatrix_h, width, height);

    // Compare with NPP
    // const int outputWidth = width - 2 * FILTER_RADIUS;
    // const int outputHeight = height - 2 * FILTER_RADIUS;
    const int outputWidth = width;
    const int outputHeight = height;
    float *nppOutputMatrix_h = (float *) malloc(outputWidth * outputHeight * sizeof(float));
    simpleConvolution(inputMatrix_h,
                      nppOutputMatrix_h,
                      filter32s,
                      {width, height},
                      {2 * FILTER_RADIUS + 1, 2 * FILTER_RADIUS + 1});

    printf("NPP\n");

    // Print the output matrix as a nice grid
    for (int row = FILTER_RADIUS; row < outputHeight - FILTER_RADIUS; row++) {
        for (int col = FILTER_RADIUS; col < outputWidth - FILTER_RADIUS; col++) {
            printf("%.2f ", nppOutputMatrix_h[row * outputWidth + col]);
        }
        printf("\n");
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        // Optionally, exit or handle the error appropriately
    }

    // Free host memory
    free(inputMatrix_h);

    return 0;
}