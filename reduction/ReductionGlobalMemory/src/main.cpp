#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    int inputSize = 64;

    float *input = (float *)malloc(inputSize * sizeof(float));
    for (int i = 0; i < inputSize; i++) {
        input[i] = 1.0f;
    }

    int result = sumReduce(input, inputSize);

    printf("Result: %d\n", result);

    free(input);

    return 0;
}