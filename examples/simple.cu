#include <stdio.h>

// Define the CUDA kernel function
__global__ void kernel(int *array) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    array[index] = index;
}

int main() {
    int num_blocks = 1;
    int num_threads_per_block = 256;
    
    // Allocate memory on the device
    int *device_array;
    cudaMalloc(&device_array, num_blocks * num_threads_per_block * sizeof(int));
    
    // Launch the kernel
    kernel<<<num_blocks, num_threads_per_block>>>(device_array);
    
    // Allocate memory on the host
    int *host_array = (int*)malloc(num_blocks * num_threads_per_block * sizeof(int));
    
    // Copy the result from the device to the host
    cudaMemcpy(host_array, device_array, num_blocks * num_threads_per_block * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print the result
    for (int i = 0; i < num_blocks * num_threads_per_block; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n");
    
    // Free the memory
    cudaFree(device_array);
    free(host_array);
    
    return 0;
}
