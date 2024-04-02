#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FP4(value) (reinterpret_cast<float4*>(&(value))[0])
#define CPU_MAX 0

// GEMM = General Matrix Multiply
// SGEMM = Single-precision General Matrix Multiply (FP32 / float)
// C = A * B
__global__ void naive_sgemm(float *A, float *B, float *C, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[m * K + k] * B[k * N + n]; // C[m * N + n] += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
    }
}

__global__ void shared_sgemm(float *A, float *B, float *C, int M, int N, int K) {
    // For a (M * K) * (K * N) GEMM, we separate M to blocks sized BM,
    // N to blocks sized BN, K to blocks sized BK.
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    // Each thread calculate TM * TN elements
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Allocate shared memory for A and B
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float Cs[TM * TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        // Load A and B from global memory to shared memory
        As[ty * BK + tx] = A[(by * BM + ty) * K + k + tx];
        Bs[ty * BK + tx] = B[(k + ty) * N + bx * BN + tx];
        __syncthreads();
        // Calculate Cs
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    Cs[i * TN + j] += As[i * BK + k] * Bs[k * BN + j]; // Cs[i * TN + j] += As[i * BK + k] * Bs[k * BN + j];
                }
            }
        }
        __syncthreads();
    }

    // Write Cs to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[(by * BM + ty + i) * N + bx * BN + tx + j] = Cs[i * TN + j]; // C[(by * BM + ty + i) * N + bx * BN + tx + j] = Cs[i * TN + j];
        }
    }
}

int main(void) {
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int repeat = 100;
    void (*sgemm) (float*, float*, float*, int, int, int) = naive_sgemm;
    float gpu_time_total = 0.0f;
    for (int i = 0; i < 15; ++i) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        dim3 block_dim(32, 32, 1);
        dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y, 1);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        float *h_A = (float*)malloc(M * K * sizeof(float));
        float *h_B = (float*)malloc(K * N * sizeof(float));
        float *h_C = (float*)malloc(M * N * sizeof(float));
        for (int j = 0; j < M * K; ++j) {
            h_A[j] = (float)rand() / RAND_MAX;
        }
        for (int j = 0; j < K * N; ++j) {
            h_B[j] = (float)rand() / RAND_MAX;
        }
        float cpu_time = 0.0f, gpu_time = 0.0f;

        // ============ cpu ============
        if (i < CPU_MAX){
            cpu_time -= (float)clock();
#pragma omp parallel for
            for (int j = 0; j < repeat; ++j) {
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < N; ++n) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += h_A[m * K + k] * h_B[k * N + n];
                        }
                        h_C[m * N + n] = sum;
                    }
                }
            }
            cpu_time += (float)clock();
            cpu_time /= CLOCKS_PER_SEC;
        } else {
            cpu_time = 0;
        }

        // ============ gpu ============
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, M * N * sizeof(float));
        cudaDeviceSynchronize();
        gpu_time -= (float)clock();
        for (int j = 0; j < repeat; ++j) {
            sgemm<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
        }
        cudaDeviceSynchronize();
        gpu_time += (float)clock();
        gpu_time /= CLOCKS_PER_SEC;
        gpu_time_total += gpu_time;

        // ============ check ============
        if (i < CPU_MAX){
            float *h_d_C = (float*)malloc(M * N * sizeof(float));
            cudaMemcpy(h_d_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            float error = 0.0f;
            for (int j = 0; j < M * N; ++j) {
                error += abs(h_d_C[j] - h_C[j]);
            }

            printf("M = %d, N = %d, K = %d, cpu_time = %f, gpu_time = %f, error = %f\n", M, N, K, cpu_time, gpu_time, error);
        } else {
            printf("M = %d, N = %d, K = %d, cpu_time = %f, gpu_time = %f\n", M, N, K, cpu_time, gpu_time);
        }
        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaDeviceReset();
    }
    printf("gpu_time_total = %f\n", gpu_time_total);
}