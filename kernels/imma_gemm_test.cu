// imma_gemm_test.cu — does int8 tensor-core (IMMA/WMMA) accumulation give the SAME
// integer result as a serial integer dot, and is it fast? This is the crux of Tier-2:
// if int8 tensor cores are bit-exact (they accumulate in int32 integer arithmetic,
// which is associative), then a quantized GEMM can use them and stay bit-identical
// across hardware — something float tensor cores cannot promise.
//
// C[M,N] (int32) = A[M,K] (int8, row-major) . B[N,K] (int8, row-major; = Bt col-major).
// Compares: WMMA tensor core vs __dp4a vs a serial integer kernel (the reference).
//
// Build: nvcc -O2 -arch=sm_75 imma_gemm_test.cu -o imma   (sm_75 T4 / sm_86 A10G both
//        have int8 tensor cores). Determinism needs no --fmad flag: it is pure integer.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// One warp computes a 16x16 int32 tile of C using int8 tensor cores.
__global__ void imma_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tilesN = N / 16;
    int tm = warp / tilesN, tn = warp % tilesN;
    if (tm * 16 >= M || tn * 16 >= N) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
    wmma::fill_fragment(c, 0);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + tm * 16 * K + k, K);
        wmma::load_matrix_sync(b, B + tn * 16 * K + k, K);   // B is [N,K] row-major -> col-major [K,N]
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + tm * 16 * N + tn * 16, c, N, wmma::mem_row_major);
}

// __dp4a: 4 int8 products per instruction, one thread per C element.
__global__ void dp4a_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N, n = idx % N;
    const int* a4 = (const int*)(A + (size_t)m * K);
    const int* b4 = (const int*)(B + (size_t)n * K);
    int acc = 0;
    for (int k = 0; k < K / 4; k++) acc = __dp4a(a4[k], b4[k], acc);
    C[(size_t)m * N + n] = acc;
}

// Serial integer reference (one thread per C element, plain loop).
__global__ void serial_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N, n = idx % N;
    const int8_t* a = A + (size_t)m * K;
    const int8_t* b = B + (size_t)n * K;
    int acc = 0;
    for (int k = 0; k < K; k++) acc += (int)a[k] * (int)b[k];
    C[(size_t)m * N + n] = acc;
}

static int diff(const int32_t* x, const int32_t* y, size_t n) { size_t d = 0; for (size_t i = 0; i < n; i++) if (x[i] != y[i]) d++; return (int)d; }

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;   // output rows (weight rows)
    int N = argc > 2 ? atoi(argv[2]) : 64;     // token batch (prefill width)
    int K = argc > 3 ? atoi(argv[3]) : 4096;   // reduction (input dim)
    int iters = argc > 4 ? atoi(argv[4]) : 50;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("== %s (CC %d.%d) ==  GEMM M=%d N=%d K=%d\n", p.name, p.major, p.minor, M, N, K);

    size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    int8_t *hA = (int8_t*)malloc(na), *hB = (int8_t*)malloc(nb);
    for (size_t i = 0; i < na; i++) hA[i] = (int8_t)((i * 37 + 11) % 255 - 127);
    for (size_t i = 0; i < nb; i++) hB[i] = (int8_t)((i * 53 + 7) % 255 - 127);
    int8_t *dA, *dB; int32_t *dC1, *dC2, *dC3;
    cudaMalloc(&dA, na); cudaMalloc(&dB, nb); cudaMalloc(&dC1, nc * 4); cudaMalloc(&dC2, nc * 4); cudaMalloc(&dC3, nc * 4);
    cudaMemcpy(dA, hA, na, cudaMemcpyHostToDevice); cudaMemcpy(dB, hB, nb, cudaMemcpyHostToDevice);

    int warps = (M / 16) * (N / 16);
    imma_gemm<<<(warps * 32 + 127) / 128, 128>>>(dA, dB, dC1, M, N, K);
    dp4a_gemm<<<(nc + 127) / 128, 128>>>(dA, dB, dC2, M, N, K);
    serial_gemm<<<(nc + 127) / 128, 128>>>(dA, dB, dC3, M, N, K);
    cudaDeviceSynchronize();
    int32_t *c1 = (int32_t*)malloc(nc * 4), *c2 = (int32_t*)malloc(nc * 4), *c3 = (int32_t*)malloc(nc * 4);
    cudaMemcpy(c1, dC1, nc * 4, cudaMemcpyDeviceToHost); cudaMemcpy(c2, dC2, nc * 4, cudaMemcpyDeviceToHost); cudaMemcpy(c3, dC3, nc * 4, cudaMemcpyDeviceToHost);
    printf("bit-identity (int32, exact):  IMMA vs serial %d diffs   dp4a vs serial %d diffs   IMMA vs dp4a %d diffs  (of %zu)\n",
           diff(c1, c3, nc), diff(c2, c3, nc), diff(c1, c2, nc), nc);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    auto time = [&](const char* nm, int which) {
        cudaEventRecord(s);
        for (int it = 0; it < iters; it++) {
            if (which == 0) imma_gemm<<<(warps * 32 + 127) / 128, 128>>>(dA, dB, dC1, M, N, K);
            else if (which == 1) dp4a_gemm<<<(nc + 127) / 128, 128>>>(dA, dB, dC2, M, N, K);
            else serial_gemm<<<(nc + 127) / 128, 128>>>(dA, dB, dC3, M, N, K);
        }
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
        double tops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        printf("  %-8s %.3f ms/GEMM   %.1f int8-TOPS\n", nm, ms, tops);
    };
    time("IMMA", 0); time("dp4a", 1); time("serial", 2);
    return 0;
}
