// Raw-PTX track, warp-spec milestone, step 2: fine-grained mbarrier pipeline. Drops
// the per-step CTA-wide __syncthreads of matmul_fp4_tma (1031k) for a full/empty
// barrier handshake -- thread0 issues TMA, all 4 warps stay consumers (no dedicated
// producer warp; CubeCL proved that loses on sm_120). full[s] flips on TMA landing;
// empty[s] flips when all 128 threads finish reading stage s, gating the producer's
// reuse. This lets consumers of other stages race ahead instead of barriering.

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
    __shared__ __align__(8) uint64_t empty[STAGES];

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
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
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
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + (arw + arow) * BKH + kb + acblk * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + (b_col[n] + nrow) * BKH + kb + bkblk * 16]);
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
        // signal stage s is drained; producer reuses it only after all 128 arrive
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) {
            uint32_t epar = ((next / STAGES) - 1) & 1;
            asm volatile("{\n\t.reg .pred p;\n\tWE:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra WE;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])), "r"(epar));
            issue(s, next);
        }
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
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map failed: %s\n", s); }
        return m;
    };
    CUtensorMap mapA = build(dA, M, Kb, BM);
    CUtensorMap mapB = build(dB, N, Kb, BN);

    dim3 grid(N / BN, M / BM), block(128);
    for (int i = 0; i < 5; i++) matmul_fp4<<<grid, block>>>(mapA, mapB, dC, N, Kf);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) matmul_fp4<<<grid, block>>>(mapA, mapB, dC, N, Kf);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double gflops = 2.0 * M * N * Kf / (ms / 1e3) / 1e9;

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) if (hC[i] != (float)Kf) wrong++;
    printf("matmul_fp4_ws: %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (%d wrong)\n",
           M, N, Kf, ms, gflops, wrong == 0 ? "PASS" : "FAIL", wrong);
    free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(2048);
    run(4096);
    run(8192);
    return 0;
}
