// Raw-PTX track, OMMA frontier: warp-specialized PINGPONG, matching CUTLASS's
// KernelTmaWarpSpecializedPingpong. 256 threads = 2 warpgroups (4 warps each); each
// warpgroup computes its OWN 128x128 output tile (adjacent 128-col halves of a
// 128x256 block) with its own smem stage buffers + TMA + barriers. Crucially each
// warpgroup syncs only its 128 threads via a NAMED barrier (bar.sync 1/2) instead of
// a CTA-wide __syncthreads, so the two warpgroups desynchronize: when WG0 stalls on
// its TMA mbarrier, WG1's mma keeps the tensor cores busy (fills the bubble). Builds
// on the 1252k 128x128 swizzled kernel (matmul_fp4_big). 2 stages to fit smem.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 128
#define BN 128
#define BK 128
#define BKH 64
#define STAGES 3   // 3-stage (98KB dynamic smem) beats 2-stage: 1386k vs 1356k @ 8192
#define ASZ (BM * BKH)   // 8192
#define BSZ (BN * BKH)   // 8192
#define WGBYTES (STAGES * ASZ + STAGES * BSZ)            // per-warpgroup buffer bytes
#define SMEM (2 * WGBYTES + 2 * STAGES * 8 + 128)        // 2 WG buffers + 2 WG barrier sets

__global__ void __launch_bounds__(256)
matmul_fp4(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
           float *C, int N, int Kfp4) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;

    uint8_t *a_s = smem + wg * WGBYTES;
    uint8_t *b_s = a_s + STAGES * ASZ;
    uint64_t *full = (uint64_t *)(smem + 2 * WGBYTES + wg * STAGES * 8);
    int sync_id = wg + 1;  // named barrier per warpgroup (1 or 2)

    int block_row = blockIdx.y * BM, block_col = blockIdx.x * (2 * BN) + wg * BN;
    int ksteps = Kfp4 / BK;

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;

    if (wtid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    asm volatile("bar.sync %0, 128;" ::"r"(sync_id));

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

    if (wtid == 0)
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
            uint32_t af[4][4], bf[8][2];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * BKH + kb + acblk * 16;
                ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BKH + kb + bkblk * 16;
                bo ^= ((bo >> 7) & 3) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]) : "r"(bd));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n;
                    float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]), "r"(bf[n][0]), "r"(bf[n][1]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));  // per-warpgroup sync, not CTA-wide
        int next = step + STAGES;
        if (wtid == 0 && next < ksteps) issue(s, next);
    }

#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2);
            int gc = block_col + b_col[n] + (lane & 3) * 2;
            C[gr * N + gc] = acc[idx][0];
            C[gr * N + gc + 1] = acc[idx][1];
            C[(gr + 8) * N + gc] = acc[idx][2];
            C[(gr + 8) * N + gc + 1] = acc[idx][3];
        }
}

void run(int sz) {
    int M = sz, N = sz, Kf = sz, Kb = Kf / 2;
    uint8_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * Kb);
    cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    cudaMemset(dA, 0x22, (size_t)M * Kb);
    cudaMemset(dB, 0x22, (size_t)N * Kb);

    auto build = [](uint8_t *p, int rows, int Kb, int boxrows) {
        CUtensorMap m;
        uint64_t gdim[2] = {(uint64_t)Kb, (uint64_t)rows};
        uint64_t gstride[1] = {(uint64_t)Kb};
        uint32_t bdim[2] = {(uint32_t)BKH, (uint32_t)boxrows};
        uint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gdim, gstride, bdim, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map failed: %s\n", s); }
        return m;
    };
    CUtensorMap mapA = build(dA, M, Kb, BM);
    CUtensorMap mapB = build(dB, N, Kb, BN);

    cudaFuncSetAttribute(matmul_fp4, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / (2 * BN), M / BM), block(256);
    for (int i = 0; i < 5; i++) matmul_fp4<<<grid, block, SMEM>>>(mapA, mapB, dC, N, Kf);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) matmul_fp4<<<grid, block, SMEM>>>(mapA, mapB, dC, N, Kf);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double gflops = 2.0 * M * N * Kf / (ms / 1e3) / 1e9;

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) if (hC[i] != (float)Kf) wrong++;
    printf("matmul_fp4_pp: %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (%d wrong)\n",
           M, N, Kf, ms, gflops, wrong == 0 ? "PASS" : "FAIL", wrong);
    free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(2048);
    run(4096);
    run(8192);
    return 0;
}
