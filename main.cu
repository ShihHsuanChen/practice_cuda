#include<stdio.h>

/* Ref: https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
The maximum x, y and z dimensions of a block are 1024, 1024 and 64, and it should be allocated
such that x × y × z ≤ 1024, which is the maximum number of threads per block.[3] Blocks can be
organized into one, two or three-dimensional grids of up to 231-1, 65,535 and 65,535 blocks in
the x, y and z dimensions respectively.[3] Unlike the maximum threads per block, there is not 
a blocks per grid limit distinct from the maximum grid dimensions.
*/
#define MAX_THREAD_X 1024;
#define MAX_THREAD_Y 1024;
#define MAX_THREAD_Z 64;
#define MAX_TOT_THREAD 1024;

// device code
// array (1D)
__global__ void kernal_add_const_1d(int *a, int c, int *o, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) o[i] = a[i] + c;
}

__global__ void kernal_add_1d(int *a1, int *a2, int *o, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) o[i] = a1[i] + a2[i];
}

__global__ void kernal_mul_1d(int* a1, int* a2, int* o, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) o[i] = a1[i] * a2[i];
}

// matrix (2D)
__global__ void kernal_add_const_2d(int* a, int c, int *o, int N, int M) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < N && j < M) o[i*M+j] = a[i*M+j] + c;
}

__global__ void kernal_add_2d(int* a1, int *a2, int *o, int N, int M) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < N && j < M) o[i*M+j] = a1[i*M+j] + a2[i*M+j];
}

__global__ void kernal_mul_2d(int* a1, int *a2, int *o, int N, int M) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < N && j < M) o[i*M+j] = a1[i*M+j] * a2[i*M+j];
}

template<typename T>
__global__ void kernal_matmul_2d(T* a1, T *a2, T *o, int N, int H, int M) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    o[i*M+j] = 0;
    if (i < N && j < M) {
        for (int k=0;k<H;k++) o[i*M+j] += a1[i*M+k] * a2[k*N+j];
    }
}

// utils
void print_arr_1d(float *a, int N) {
    printf("[");
    for (int i=0;i<N-1;i++) printf("%f, ", a[i]);
    printf("%f]\n", a[N-1]);
}

void print_arr_2d(float *a, int N, int M) {
    printf("[");
    for (int i=0;i<N-1;i++) {
        print_arr_1d(&a[i*M], M);
        printf(" ");
    }
    printf("[");
    for (int j=0;j<M-1;j++) printf("%f, ", a[(N-1)*M+j]);
    printf("%f]]\n", a[N*M-1]);
}

float* create_device_arr_1d(int N) {
    float *d_arr;
    cudaMalloc((void**)&d_arr, sizeof(float)*N);
    return d_arr;
}

float* create_device_arr_1d(int N, float* h_arr) {
    float *d_arr = create_device_arr_1d(N);
    cudaMemcpy(d_arr, h_arr, sizeof(float)*N, cudaMemcpyHostToDevice);
    return d_arr;
}

float* create_device_arr_2d(int N, int M) {
    return create_device_arr_1d(N*M);
}

float* create_device_arr_2d(int N, int M, float* h_arr) {
    return create_device_arr_1d(N*M, h_arr);
}

// main
int main() {
    int N = 3;
    int M = 3;
    // define array 1
    float h_arr1[N*M] = {
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
    };
    print_arr_2d(h_arr1, N, M); // TODO

    // define array 2
    float h_arr2[M*N] = {
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
    };
    print_arr_2d(h_arr2, M, N);

    // define output
    float h_arr_out[N*N];

    // define device pointers and copy host data to device
    float *d_arr1 = create_device_arr_2d(N, M, h_arr1);
    float *d_arr2 = create_device_arr_2d(M, N, h_arr2);
    float *d_arr_out = create_device_arr_2d(N, N); // NM * MN = NN

    // define kernal
    dim3 dim_grid(1);
    dim3 dim_block(M, N);

    // run kernal
    kernal_matmul_2d<<<dim_grid, dim_block>>>(d_arr1, d_arr2, d_arr_out, N, M, N);

    // copy from device to host
    cudaMemcpy(h_arr_out, d_arr_out, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    print_arr_2d(h_arr_out, N, N);
    return 0;
}
