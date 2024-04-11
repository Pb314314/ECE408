// MP 1
#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);      // return the wbarg_t, which contain the input file, output file, expected file and type

    wbTime_start(Generic, "Importing data and creating memory on host");
    // wbArg_getInputFile(args, 0) return the pointer to the first file path
    // wbImport(file path, &inputLength), input file path, return float array, set input length
    // need two input files for this mp. set inputlength using 
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);     // first input float array
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);     // second input float array
    hostOutput = (float *) malloc(inputLength * sizeof(float));                     // output array of float
    wbTime_stop(Generic, "Importing data and creating memory on host");

    // wbLog(kind, message, number);information in log section
    wbLog(TRACE, "The input length is ", inputLength);      

    // wbTime_start(kind, message);information in timer section
	wbTime_start(GPU, "Allocating GPU memory.");
    // section for allocating GPU memory

    // cudaMalloc(void** devPtr, size_t size); 
    // first: a pointer to allocated device memory, second: number of bytes to allocate
    // first pointer need to use reference, cudaMalloc will modify the pointer to point to the allocated memory on the GPU
    // also need to cast pointer to void** to take the address of any type pointer
    // sizeof return the number of byte per float(return type size_t)
    cudaMalloc((void**)deviceInput1, sizeof(float) * inputLength);
    cudaMalloc((void**)deviceInput2, sizeof(float) * inputLength);
    cudaMalloc((void**)deviceOutput, sizeof(float) * inputLength);
    wbTime_stop(GPU, "Allocating GPU memory.");


    wbTime_start(GPU, "Copying input memory to the GPU.");
    // section for copy input memory from CPU to GPU
    // cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    // don't need explicit cast float* to void*, can implicit cast
    cudaMemcpy(deviceInput1, hostInput1, sizeof(float) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(float) * inputLength, cudaMemcpyHostToDevice);
    
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here


    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);
    // free float array
    free(hostInput1);       
    free(hostInput2);       
    free(hostOutput);

    return 0;
}

