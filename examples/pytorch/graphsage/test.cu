#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

__global__ void bubbleSortSharedMemory(int *arr, int n) {
    // Allocate shared memory
    extern __shared__ int shared_arr[];

    // Thread index within block
    int tid = threadIdx.x;

    // Load global memory data into shared memory
    if (tid < n) {
        shared_arr[tid] = arr[tid];
    }
    
    // Synchronize to make sure all data is loaded into shared memory
    __syncthreads();

    // Bubble sort in shared memory
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (shared_arr[j] < shared_arr[j + 1]) {
                // Swap elements
                int temp = shared_arr[j];
                shared_arr[j] = shared_arr[j + 1];
                shared_arr[j + 1] = temp;
            }
        }
        // Synchronize threads to ensure all threads have the correct data after each iteration
        __syncthreads();
    }

    // Write back sorted data to global memory
    if (tid < n) {
        arr[tid] = shared_arr[tid];
    }
}

int main() {
    const int n = 5;
    int h_arr[n] = {5, 3, 8, 4, 1};

    // Allocate memory on the device
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and n threads, using shared memory
    bubbleSortSharedMemory<<<1, n, n * sizeof(int)>>>(d_arr, n);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy sorted array back to host
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}

