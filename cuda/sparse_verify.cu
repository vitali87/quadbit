// Raw-PTX track, SPARSE matmul: 2:4-sparse block-scaled FP4 with mma.sp (m16n8k128),
// the 2x-throughput path. Each mma.sp covers 128 LOGICAL K (64 nonzero, A compressed)
// vs dense m16n8k64, so effective GFLOP/s (2*M*N*Klogical/time) should ~2x dense and
// beat CUTLASS dense (1504k). 4-warp block, 64x128 tile, 2x8 warp tile, cp.async
// double-buffer. All-ones => every output == Klogical/2 (the 2:4 nonzero count).
// A compressed: 32 bytes/row (64 nonzero fp4). B full: 64 bytes/row (128 fp4). ue4m3
// unit scales (0x38), scale_vec::4X. Metadata constant 0x44444444 (all-ones-agnostic).

#include <cstdio>
#include <cstdint>

#define BM 64
#define BN 128
#define BKL 128          // logical K per step (one mma.sp K-slice)
#define AROWB 32         // compressed A bytes/row (64 nonzero fp4)
#define BROWB 64         // full B bytes/row (128 fp4)
#define WARPS 4
#define ASZ (BM * AROWB)
#define BSZ (BN * BROWB)

__device__ __forceinline__ void stage(const uint8_t *A, const uint8_t *B, uint8_t *a_s, uint8_t *b_s,
                                       int buf, int step, int block_row, int block_col, int KAb, int KBb, int tid) {
#pragma unroll
    for (int c = tid; c < ASZ / 16; c += 128) {
        int row = c / (AROWB / 16), colb = (c % (AROWB / 16)) * 16;
        uint32_t da = __cvta_generic_to_shared(&a_s[buf * ASZ + row * AROWB + colb]);
        const void *ga = &A[(block_row + row) * KAb + step * AROWB + colb];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(da), "l"(ga));
    }
#pragma unroll
    for (int c = tid; c < BSZ / 16; c += 128) {
        int row = c / (BROWB / 16), colb = (c % (BROWB / 16)) * 16;
        uint32_t db = __cvta_generic_to_shared(&b_s[buf * BSZ + row * BROWB + colb]);
        const void *gb = &B[(block_col + row) * KBb + step * BROWB + colb];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(db), "l"(gb));
    }
}

__global__ void __launch_bounds__(128) matmul_sp(const uint8_t *A, const uint8_t *B, float *C, int N, int Klog) {
    __shared__ __align__(16) uint8_t a_s[2 * ASZ];
    __shared__ __align__(16) uint8_t b_s[2 * BSZ];

    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp >> 1, wn = warp & 1;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    int KAb = Klog / 4;            // compressed A bytes per global row (64 nonzero/128 logical = /4)
    int KBb = Klog / 2;            // full B bytes per global row
    int ksteps = Klog / BKL;

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_row0 = (wm * 2) * 16, a_row1 = (wm * 2 + 1) * 16;
    int b_col[8];
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = (wn * 8 + j) * 8;

    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x38383838u, sb = 0x38383838u;  // ue4m3 unit scales
    uint32_t meta = 0x44444444u;
    uint16_t z = 0;

    stage(A, B, a_s, b_s, 0, 0, block_row, block_col, KAb, KBb, tid);
    asm volatile("cp.async.commit_group;");

    for (int step = 0; step < ksteps; step++) {
        int cur = step & 1, aoff = cur * ASZ, boff = cur * BSZ;
        if (step + 1 < ksteps) {
            stage(A, B, a_s, b_s, (step + 1) & 1, step + 1, block_row, block_col, KAb, KBb, tid);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        uint32_t af[2][4], bf[8][4];
#pragma unroll
        for (int m = 0; m < 2; m++) {
            int arw = m == 0 ? a_row0 : a_row1;
            uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + (arw + arow) * AROWB + acblk * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3]) : "r"(ad));
        }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t bd = __cvta_generic_to_shared(&b_s[boff + (b_col[n] + nrow) * BROWB + bsub * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
        }
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int idx = m * 8 + n;
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]),
                      "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                      "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                      "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
            }
        __syncthreads();
    }

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

static float decode_fp4(uint8_t n) {
    int sign = (n >> 3) & 1, exp = (n >> 1) & 3, man = n & 1;
    float v = (exp == 0) ? (man ? 0.5f : 0.f) : ((float)(1 << (exp - 1)) * (1.f + 0.5f * man));
    return sign ? -v : v;
}

void run(int sz) {
    int M = sz, N = sz, Klog = sz;
    int KAb = Klog / 4, KBb = Klog / 2;  // compressed A bytes (64 nz/128), full B bytes
    uint8_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * KAb);
    cudaMalloc(&dB, (size_t)N * KBb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));

    // metadata 0x44444444 selects logical positions 0,1 of each group-of-4 as nonzero.
    // compressed byte p of a row holds the nonzero nibbles at logical 4p (low), 4p+1 (high).
    uint8_t *hAc = (uint8_t *)malloc((size_t)M * KAb), *hB = (uint8_t *)malloc((size_t)N * KBb);
    uint8_t *hAlog = (uint8_t *)malloc((size_t)M * Klog);  // nibble per logical k (0 where masked)
    uint32_t st = 0x5A5Au;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (int i = 0; i < M; i++)
        for (int p = 0; p < KAb; p++) {
            uint8_t lo = rnd() & 0xf, hi = rnd() & 0xf;       // two nonzero nibbles
            hAc[(size_t)i * KAb + p] = lo | (hi << 4);
            hAlog[(size_t)i * Klog + 4 * p + 0] = lo;          // logical pos 0
            hAlog[(size_t)i * Klog + 4 * p + 1] = hi;          // logical pos 1
            hAlog[(size_t)i * Klog + 4 * p + 2] = 0;           // pos 2,3 masked
            hAlog[(size_t)i * Klog + 4 * p + 3] = 0;
        }
    for (size_t i = 0; i < (size_t)N * KBb; i++) hB[i] = (uint8_t)rnd();
    cudaMemcpy(dA, hAc, (size_t)M * KAb, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * KBb, cudaMemcpyHostToDevice);

    dim3 grid(N / BN, M / BM), block(128);
    matmul_sp<<<grid, block>>>(dA, dB, dC, N, Klog);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // host ref: out[i][j] = sum_k decode(Alog[i][k]) * decode(B[j][k]) over nonzero k (pos 0,1)
    int wrong = 0; float maxrel = 0.f;
    for (int i = 0; i < M && wrong < 8; i++)
        for (int j = 0; j < N; j++) {
            float ref = 0.f;
            for (int k = 0; k < Klog; k++) {
                if ((k & 3) >= 2) continue;  // masked
                uint8_t bn = hB[(size_t)j * KBb + k / 2];
                float bv = decode_fp4(k & 1 ? bn >> 4 : bn & 0xf);
                ref += decode_fp4(hAlog[(size_t)i * Klog + k]) * bv;
            }
            float got = hC[(size_t)i * N + j];
            float rel = fabsf(got - ref) / (fabsf(ref) + 1.f);
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-2f) { if (wrong < 4) printf("  [%d][%d] got %.2f ref %.2f\n", i, j, got, ref); wrong++; }
        }
    printf("sparse_verify (2:4 metadata 0x44, random A/B, unit scales): %dx%dx%d  %s (%d wrong, maxrel %.4f)\n",
           M, N, Klog, wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    free(hAc); free(hB); free(hAlog); free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(256);
    return 0;
}
