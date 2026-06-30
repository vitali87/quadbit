// Raw-PTX track: 2:4-sparse FP4 matmul, traffic-optimal tiling. The pingpong kernel is
// 100% memory-bound (mem-only TMA probe == full kernel time): sparse computes 2x faster
// than dense but loads the same B bytes, so DRAM/L2 bandwidth is the wall. This kernel
// minimizes B traffic: the 2 warpgroups (acc-capped at 128x128 each) stack in M to form a
// 256x128 CTA tile sharing ONE B column-panel, instead of 128x256 side-by-side. With B
// at 2x the bytes/row of compressed A, BM_eff=256/BN_eff=128 is the traffic optimum
// (20% less total, B DRAM requests halved). Compute is fully hidden so the pingpong
// desync is dropped for a simpler CTA-wide pipeline; shared B also frees smem => deeper
// staging. All-ones => out == Klog/2.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 128           // rows per warpgroup tile
#define BN 128           // cols per CTA (shared by both warpgroups)
#define BKL 128
#define AROWB 32
#define BROWB 64
#define STAGES 5
#define ASZ (BM * AROWB) // 4096 per warpgroup A-half
#define BSZ (BN * BROWB) // 8192 shared B
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + STAGES * 8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          float *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;

    uint8_t *a_s = smem + wg * STAGES * ASZ;           // this warpgroup's A-half
    uint8_t *b_s = smem + 2 * STAGES * ASZ;            // shared B
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);

    int block_row = blockIdx.y * (2 * BM) + wg * BM;   // 256-row CTA, this WG's 128
    int a_load_row = blockIdx.y * (2 * BM);            // CTA's first row (both A-halves)
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

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    // producer (tid 0): 2 A-halves + 1 shared B per stage
    auto issue = [&](int s, int step) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])),
                     "l"(&mapB), "r"(step * BROWB), "r"(block_col), "r"(bar));
    };

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

        // MEM-ONLY: no LDSM/mma; touch one byte so loop time = pure TMA pipeline time.
        acc[0][0] += (float)a_s[s * ASZ + wtid] + (float)b_s[s * BSZ + wtid];
        __syncthreads();
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) issue(s, next);
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

static void mk(const char *tag, CUtensorMap *m, uint8_t *p, int inner, int outer, int boxin, int boxout) {
    uint64_t gdim[2] = {(uint64_t)inner, (uint64_t)outer};
    uint64_t gstride[1] = {(uint64_t)inner};
    uint32_t bdim[2] = {(uint32_t)boxin, (uint32_t)boxout};
    uint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gdim, gstride, bdim, estride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map[%s] failed: %s\n", tag, s); }
}

void run(int sz) {
    int M = sz, N = sz, Klog = sz, KAb = Klog / 4, KBb = Klog / 2;
    uint8_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * KAb);
    cudaMalloc(&dB, (size_t)N * KBb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    cudaMemset(dA, 0x22, (size_t)M * KAb);
    cudaMemset(dB, 0x22, (size_t)N * KBb);

    alignas(64) CUtensorMap mapA, mapB;
    mk("A", &mapA, dA, KAb, M, AROWB, BM);
    mk("B", &mapB, dB, KBb, N, BROWB, BN);

    cudaFuncSetAttribute(matmul_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / BN, M / (2 * BM)), block(256);
    for (int i = 0; i < 5; i++) matmul_sp<<<grid, block, SMEM>>>(mapA, mapB, dC, M, N, Klog);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) matmul_sp<<<grid, block, SMEM>>>(mapA, mapB, dC, M, N, Klog);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double gflops = 2.0 * M * N * Klog / (ms / 1e3) / 1e9;

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) if (hC[i] != (float)(Klog / 2)) wrong++;
    printf("matmul_sp_bm256_mem (TMA only, shared-B): %dx%dx%d  %.3f ms  %.1f GFLOP/s  %s (out[0]=%.0f exp %d)\n",
           M, N, Klog, ms, gflops, wrong == 0 ? "PASS" : "FAIL", hC[0], Klog / 2);
    free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(2048);
    run(4096);
    run(8192);
    return 0;
}
