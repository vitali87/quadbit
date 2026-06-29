// Raw-PTX track, milestone 2: full FP4 matmul hand-written in CUDA, mirroring the
// best CubeCL kernel (ldm2): 4-warp block (64x128 tile), 2x8 warp tile (16 f32
// accumulators), shared staging, ldmatrix loads, K-loop. All-ones FP4 + unit scales
// => every output == K, and we time GFLOP/s to compare against ldm2's ~497k@4096.
// Real per-block scales + tuning come next; this validates the full raw mainloop at
// scale and gives the first raw performance number.

#include <cstdio>
#include <cstdint>

#define BM 64
#define BN 128
#define BK 128
#define BKH 64   // BK/2 bytes per staged row
#define WARPS 4

__global__ void __launch_bounds__(128) matmul_fp4(const uint8_t *A, const uint8_t *B, float *C, int N, int Kfp4) {
    __shared__ __align__(16) uint8_t a_s[BM * BKH];
    __shared__ __align__(16) uint8_t b_s[BN * BKH];

    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp >> 1, wn = warp & 1;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    int Kb = Kfp4 / 2;            // bytes per global row
    int ksteps = Kfp4 / BK;

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    int a_row0 = (wm * 2) * 16, a_row1 = (wm * 2 + 1) * 16;
    int b_col[8];
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = (wn * 8 + j) * 8;

    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;

    for (int step = 0; step < ksteps; step++) {
        int kvec = step * BKH;
        for (int i = tid; i < BM * BKH; i += 128) a_s[i] = A[(block_row + i / BKH) * Kb + kvec + i % BKH];
        for (int i = tid; i < BN * BKH; i += 128) b_s[i] = B[(block_col + i / BKH) * Kb + kvec + i % BKH];
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < BK / 64; ks++) {
            int kb = ks * 32;
            uint32_t af[2][4], bf[8][2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                int arw = m == 0 ? a_row0 : a_row1;
                uint32_t ad = __cvta_generic_to_shared(&a_s[(arw + arow) * BKH + kb + acblk * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                uint32_t bd = __cvta_generic_to_shared(&b_s[(b_col[n] + nrow) * BKH + kb + bkblk * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]) : "r"(bd));
            }
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = m * 8 + n;
                    float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]), "r"(bf[n][0]), "r"(bf[n][1]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        __syncthreads();
    }

    // store per the m16n8 f32 accumulator layout
#pragma unroll
    for (int m = 0; m < 2; m++) {
        int arw = m == 0 ? a_row0 : a_row1;
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = m * 8 + n;
            int gr = block_row + arw + (lane >> 2);
            int gc = block_col + b_col[n] + (lane & 3) * 2;
            C[gr * N + gc] = acc[idx][0];
            C[gr * N + gc + 1] = acc[idx][1];
            C[(gr + 8) * N + gc] = acc[idx][2];
            C[(gr + 8) * N + gc + 1] = acc[idx][3];
        }
    }
}

int main() {
    int M = 4096, N = 4096, Kf = 4096;
    int Kb = Kf / 2;
    uint8_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * Kb);
    cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    cudaMemset(dA, 0x22, (size_t)M * Kb);
    cudaMemset(dB, 0x22, (size_t)N * Kb);

    dim3 grid(N / BN, M / BM), block(128);
    for (int i = 0; i < 5; i++) matmul_fp4<<<grid, block>>>(dA, dB, dC, N, Kf);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) matmul_fp4<<<grid, block>>>(dA, dB, dC, N, Kf);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double gflops = 2.0 * M * N * Kf / (ms / 1e3) / 1e9;

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) if (hC[i] != (float)Kf) wrong++;
    printf("matmul_fp4: %dx%dx%d  out[0]=%.1f (expected %d)  %.3f ms  %.1f GFLOP/s  %s (%d wrong)\n",
           M, N, Kf, hC[0], Kf, ms, gflops, wrong == 0 ? "PASS" : "FAIL", wrong);
    return 0;
}
