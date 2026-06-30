// RESULT (sm_120): DEAD-END. ptxas advises `.multicast::cluster` "is expected to have
// substantially reduced performance" on sm_120a and belongs on sm_90a/100a/101a --
// TMA multicast is datacenter-gated on this consumer/pro Blackwell (same as no-tcgen05,
// no-CUTLASS-sparse-path). This 2-CTA cross-CTA pipeline also deadlocks (cluster-barrier
// ordering between arm/multicast/consume is CUTLASS-grade and untested). Abandoned: even
// if debugged it would be SLOWER on sm_120. Confirms 2012k (single-CTA) is the hw ceiling.
//
// Raw-PTX track: beat the single-CTA bandwidth roofline with a 2-CTA CLUSTER + TMA
// B-multicast. Two CTAs of a cluster (same columns, adjacent row-blocks = 512 rows total)
// share ONE B column-panel: the leader issues a multicast TMA that delivers B into BOTH
// CTAs' smem and signals both mbarriers, so B DRAM traffic halves vs the per-CTA load
// (B is ~half of A+B traffic => ~25% less total on this memory-bound kernel). A is loaded
// per-CTA (own rows). Cross-CTA ordering via cluster barriers. All-ones validation: out
// must equal Klog/2. This is the CUTLASS multicast scheme on sm_120.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 128
#define BN 128
#define BKL 128
#define AROWB 32
#define BROWB 64
#define STAGES 3
#define ASZ (BM * AROWB)
#define BSZ (BN * BROWB)
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + STAGES * 8 + 128)

__global__ void __cluster_dims__(1, 2, 1) __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          float *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint32_t ctarank; asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(ctarank));
    bool leader = (ctarank == 0);

    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);

    int block_row = blockIdx.y * (2 * BM) + wg * BM;
    int a_load_row = blockIdx.y * (2 * BM);
    int block_col = blockIdx.x * BN;
    int ksteps = Klog / BKL;

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
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
    uint32_t sa = 0x38383838u, sb = 0x38383838u, meta = 0x44444444u;
    uint16_t z = 0;

    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
    asm volatile("fence.proxy.async.shared::cta;");
    asm volatile("barrier.cluster.arrive;"); asm volatile("barrier.cluster.wait;");

    // arm full[s] (own A + multicast B portion) and issue own 2 A halves
    auto armA = [&](int s, int step) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row + BM), "r"(bar));
    };
    // leader multicasts B into BOTH CTAs' b_s and signals BOTH full[s] (mask 0b11)
    auto bMC = [&](int s, int step) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                     " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])),
                     "l"(&mapB), "r"(step * BROWB), "r"(block_col), "r"(bar), "h"((uint16_t)0x3));
    };

    // prologue
    for (int s = 0; s < STAGES; s++) {
        if (tid == 0 && s < ksteps) armA(s, s);
        asm volatile("barrier.cluster.arrive;"); asm volatile("barrier.cluster.wait;");
        if (leader && tid == 0 && s < ksteps) bMC(s, s);
    }

    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\n\tWAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra WAIT;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));

        int aoff = s * ASZ, boff = s * BSZ;
        uint32_t af[4][4], bf[8][4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + (a_rowt[mt] + arow) * AROWB + acblk * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
        }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            uint32_t bd = __cvta_generic_to_shared(&b_s[boff + (b_col[n] + nrow) * BROWB + bsub * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int idx = mt * 8 + n;
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                      "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                      "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                      "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
            }
        // both CTAs done reading b_s[s]; gate reissue across the cluster
        asm volatile("barrier.cluster.arrive;"); asm volatile("barrier.cluster.wait;");
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) armA(s, next);
        asm volatile("barrier.cluster.arrive;"); asm volatile("barrier.cluster.wait;");
        if (leader && tid == 0 && next < ksteps) bMC(s, next);
    }

#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2);
            int gc = block_col + b_col[n] + (lane & 3) * 2;
            C[gr * N + gc] = acc[idx][0]; C[gr * N + gc + 1] = acc[idx][1];
            C[(gr + 8) * N + gc] = acc[idx][2]; C[(gr + 8) * N + gc + 1] = acc[idx][3];
        }
}

static void mk(const char *t, CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo) {
    uint64_t gd[2] = {(uint64_t)inner, (uint64_t)outer}; uint64_t gs[1] = {(uint64_t)inner};
    uint32_t bd[2] = {(uint32_t)bi, (uint32_t)bo}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map[%s] fail: %s\n", t, s); }
}

void run(int sz) {
    int M = sz, N = sz, Klog = sz, KAb = Klog / 4, KBb = Klog / 2;
    uint8_t *dA, *dB; float *dC;
    cudaMalloc(&dA, (size_t)M * KAb); cudaMalloc(&dB, (size_t)N * KBb); cudaMalloc(&dC, (size_t)M * N * 4);
    cudaMemset(dA, 0x22, (size_t)M * KAb); cudaMemset(dB, 0x22, (size_t)N * KBb);
    alignas(64) CUtensorMap mapA, mapB; mk("A", &mapA, dA, KAb, M, AROWB, BM); mk("B", &mapB, dB, KBb, N, BROWB, BN);
    cudaFuncSetAttribute(matmul_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / BN, M / (2 * BM)), block(256);
    for (int i = 0; i < 5; i++) matmul_sp<<<grid, block, SMEM>>>(mapA, mapB, dC, M, N, Klog);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e); int it = 20;
    cudaEventRecord(s); for (int i = 0; i < it; i++) matmul_sp<<<grid, block, SMEM>>>(mapA, mapB, dC, M, N, Klog);
    cudaEventRecord(e); cudaEventSynchronize(e); float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= it;
    double gf = 2.0 * M * N * Klog / (ms / 1e3) / 1e9;
    float *hC = (float *)malloc((size_t)M * N * 4); cudaMemcpy(hC, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
    int wrong = 0; for (size_t i = 0; i < (size_t)M * N; i++) if (hC[i] != (float)(Klog / 2)) wrong++;
    printf("matmul_sp_cluster (2-CTA B-multicast): %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (out[0]=%.0f exp %d)\n",
           M, N, Klog, ms, gf, wrong == 0 ? "PASS" : "FAIL", hC[0], Klog / 2);
    free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
int main() { run(2048); run(4096); run(8192); return 0; }
