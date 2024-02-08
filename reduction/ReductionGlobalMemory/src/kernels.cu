__global__ void sumReduceKernel(float *input, float *output) {
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        // Only threads in even positions participate
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void sumReduceConvergentKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (i < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

int sumReduce(float *input, int size) {
    float *d_input, *d_output;
    int output;

    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int numActiveThreads = size / 2;

    dim3 dimBlock(numActiveThreads, 1, 1);
    dim3 dimGrid(numActiveThreads / 1024 + 1, 1, 1);

    sumReduceKernel<<<dimGrid, dimBlock>>>(d_input, d_output);
    cudaDeviceSynchronize();

    sumReduceConvergentKernel<<<dimGrid, dimBlock>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}