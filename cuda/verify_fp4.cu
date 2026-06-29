// Raw-PTX track, OMMA frontier: add smem SWIZZLE on top of the 1031k TMA kernel to
// kill the 4-way ldmatrix bank conflict (unswizzled 64B rows put A rows 0,2,4,6 on
// banks 0-3). CUTLASS uses Swizzle<2,4,3> == CU_TENSOR_MAP_SWIZZLE_64B; TMA writes
// the tile swizzled, so ldmatrix applies the matching XOR (off ^= ((off>>7)&3)<<4)
// to its in-tile byte offset. Stage bases are 512B-aligned so the swizzle phase is
// consistent per stage. NOTE: all-ones data can't validate swizzle correctness
// (every byte 0x22), only perf + no-crash; real-data check is separate.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 64
#define BN 128
#define BK 128
#define BKH 64
#define STAGES 3
#define ASZ (BM * BKH)
#define BSZ (BN * BKH)

__global__ void __launch_bounds__(128)
matmul_fp4(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
           float *C, int N, int Kfp4) {
    __shared__ __align__(128) uint8_t a_s[STAGES * ASZ];
    __shared__ __align__(128) uint8_t b_s[STAGES * BSZ];
    __shared__ __align__(8) uint64_t full[STAGES];

    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp >> 1, wn = warp & 1;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
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

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    // helper: issue the two TMA loads (A,B) for global kstep into ring stage `s`
    auto issue = [&](int s, int kstep) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&a_s[s * ASZ])),
                     "l"(&mapA), "r"(kstep * BKH), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])),
                     "l"(&mapB), "r"(kstep * BKH), "r"(block_col), "r"(bar));
    };

    // prologue: prefetch first STAGES stages
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < ksteps) issue(s, s);

    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\n\tWAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra WAIT;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));

        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int ks = 0; ks < BK / 64; ks++) {
            int kb = ks * 32;
            uint32_t af[2][4], bf[8][2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                int arw = m == 0 ? a_row0 : a_row1;
                int ao = (arw + arow) * BKH + kb + acblk * 16;
                ao ^= ((ao >> 7) & 3) << 4;  // 64B swizzle, matches TMA
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BKH + kb + bkblk * 16;
                bo ^= ((bo >> 7) & 3) << 4;  // 64B swizzle, matches TMA
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
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
        __syncthreads();  // all warps done reading stage s before TMA refills it
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) issue(s, next);
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

// e2m1 fp4 nibble -> f32 value: sign(1) exp(2) man(1).
static float decode_fp4(uint8_t n) {
    int sign = (n >> 3) & 1, exp = (n >> 1) & 3, man = n & 1;
    float v = (exp == 0) ? (man ? 0.5f : 0.f) : ((float)(1 << (exp - 1)) * (1.f + 0.5f * man));
    return sign ? -v : v;
}

void run(int sz) {
    int M = sz, N = sz, Kf = sz, Kb = Kf / 2;
    uint8_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * Kb);
    cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));

    // random FP4 bytes (any byte is a valid pair of e2m1 nibbles); unit scales for now
    uint8_t *hA = (uint8_t *)malloc((size_t)M * Kb), *hB = (uint8_t *)malloc((size_t)N * Kb);
    uint32_t st = 0x12345678u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return (uint8_t)st; };
    for (size_t i = 0; i < (size_t)M * Kb; i++) hA[i] = rnd();
    for (size_t i = 0; i < (size_t)N * Kb; i++) hB[i] = rnd();
    cudaMemcpy(dA, hA, (size_t)M * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * Kb, cudaMemcpyHostToDevice);

    auto build = [](uint8_t *p, int rows, int Kb, int boxrows) {
        CUtensorMap m;
        uint64_t gdim[2] = {(uint64_t)Kb, (uint64_t)rows};
        uint64_t gstride[1] = {(uint64_t)Kb};
        uint32_t bdim[2] = {(uint32_t)BKH, (uint32_t)boxrows};
        uint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gdim, gstride, bdim, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map failed: %s\n", s); }
        return m;
    };
    CUtensorMap mapA = build(dA, M, Kb, BM);
    CUtensorMap mapB = build(dB, N, Kb, BN);

    dim3 grid(N / BN, M / BM), block(128);
    matmul_fp4<<<grid, block>>>(mapA, mapB, dC, N, Kf);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // host reference: C[m][n] = sum_k decode(A[m][k]) * decode(B[n][k]); byte p holds
    // fp4 element 2p in low nibble, 2p+1 in high nibble.
    int wrong = 0;
    float maxerr = 0.f;
    for (int m = 0; m < M && wrong < 10; m++)
        for (int n = 0; n < N; n++) {
            float ref = 0.f;
            for (int p = 0; p < Kb; p++) {
                uint8_t ba = hA[(size_t)m * Kb + p], bb = hB[(size_t)n * Kb + p];
                ref += decode_fp4(ba & 0xf) * decode_fp4(bb & 0xf);
                ref += decode_fp4(ba >> 4) * decode_fp4(bb >> 4);
            }
            float got = hC[(size_t)m * N + n];
            float e = got - ref; if (e < 0) e = -e;
            if (e > maxerr) maxerr = e;
            if (e > 1e-1f) {
                if (wrong < 5) printf("  mismatch [%d][%d]: got %.1f ref %.1f\n", m, n, got, ref);
                wrong++;
            }
        }
    printf("verify_fp4 (random values, unit scales): %dx%dx%d  %s (%d wrong, maxerr %.3f)\n",
           M, N, Kf, wrong == 0 ? "PASS" : "FAIL", wrong, maxerr);
    free(hA); free(hB); free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(256);
    run(512);
    return 0;
}
