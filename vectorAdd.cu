#include <stdio.h>

// CUDA Kernel for Vector Addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    printf("Hello World from GPU!\n");
}

int main() {
    int n = 1024; // Size of the vectors
    int *a, *b, *c; // Host copies of a, b, c
    int *d_a, *d_b, *d_c; // Device copies of a, b, c
    int size = n * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); 
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Setup input values
    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i*i;
    }

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch vectorAdd() kernel on GPU with N blocks
    vectorAdd<<<(n + 255)/256, 256>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    return 0;
}
