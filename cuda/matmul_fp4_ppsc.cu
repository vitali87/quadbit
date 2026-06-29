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
#define STAGES 3   // 3 data stages: deep data prefetch dominates (2 stages -> 674k)
#define ASZ (BM * BKH)   // 8192
#define BSZ (BN * BKH)   // 8192
#define WGBYTES (STAGES * ASZ + STAGES * BSZ)            // per-warpgroup buffer bytes
#define SFSM 1024                                        // per-WG SF smem: 128 A + 128 B u32
#define SMEM (2 * WGBYTES + 2 * STAGES * 8 + 2 * SFSM + 128)

__global__ void __launch_bounds__(256)
matmul_fp4(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
           float *C, int N, int Kfp4, const uint8_t *SFA, const uint8_t *SFB, int Ksf) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;

    uint8_t *a_s = smem + wg * WGBYTES;
    uint8_t *b_s = a_s + STAGES * ASZ;
    uint64_t *full = (uint64_t *)(smem + 2 * WGBYTES + wg * STAGES * 8);
    uint32_t *sfa_s = (uint32_t *)(smem + 2 * WGBYTES + 2 * STAGES * 8 + wg * SFSM);  // [128] A row scales
    uint32_t *sfb_s = sfa_s + 128;                                                    // [128] B col scales
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
    uint16_t z = 0;
    // scale-register row/col this lane owns (scale_probe layout, bid=tid=0):
    // A lane (lane&3)<2 holds row (lane>>2)+8*(lane&1); B lane (lane&3)==0 holds col lane>>2.
    int sa_lrow = (lane >> 2) + 8 * (lane & 1);
    int sb_lcol = lane >> 2;

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

        // stage this step's SF tile (128 A-rows + 128 B-cols, one u32 = the step's 4
        // K-blocks) into smem once per WG, killing the per-lane redundancy.
        sfa_s[wtid] = *(const uint32_t *)&SFA[(block_row + wtid) * Ksf + step * 4];
        sfb_s[wtid] = *(const uint32_t *)&SFB[(block_col + wtid) * Ksf + step * 4];
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
        uint32_t saC[4], sbC[8];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) saC[mt] = sfa_s[a_rowt[mt] + sa_lrow];
#pragma unroll
        for (int n = 0; n < 8; n++) sbC[n] = sfb_s[b_col[n] + sb_lcol];

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
                    uint32_t sa = ks ? (saC[mt] >> 16) : (saC[mt] & 0xffffu);
                    uint32_t sb = ks ? (sbC[n] >> 16) : (sbC[n] & 0xffffu);
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
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));  // per-warpgroup sync (data + SF reuse)
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

static float decode_fp4(uint8_t n) {
    int sign = (n >> 3) & 1, exp = (n >> 1) & 3, man = n & 1;
    float v = (exp == 0) ? (man ? 0.5f : 0.f) : ((float)(1 << (exp - 1)) * (1.f + 0.5f * man));
    return sign ? -v : v;
}

void run(int sz, bool verify) {
    int M = sz, N = sz, Kf = sz, Kb = Kf / 2, Ksf = Kf / 32;
    uint8_t *dA, *dB, *dSFA, *dSFB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * Kb);
    cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dSFA, (size_t)M * Ksf);
    cudaMalloc(&dSFB, (size_t)N * Ksf);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));

    uint8_t *hA = (uint8_t *)malloc((size_t)M * Kb), *hB = (uint8_t *)malloc((size_t)N * Kb);
    uint8_t *hSFA = (uint8_t *)malloc((size_t)M * Ksf), *hSFB = (uint8_t *)malloc((size_t)N * Ksf);
    uint32_t st = 0xBEEF01u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (size_t i = 0; i < (size_t)M * Kb; i++) hA[i] = verify ? (uint8_t)rnd() : 0x22;
    for (size_t i = 0; i < (size_t)N * Kb; i++) hB[i] = verify ? (uint8_t)rnd() : 0x22;
    for (size_t i = 0; i < (size_t)M * Ksf; i++) hSFA[i] = verify ? (125 + rnd() % 5) : 127;
    for (size_t i = 0; i < (size_t)N * Ksf; i++) hSFB[i] = verify ? (125 + rnd() % 5) : 127;
    cudaMemcpy(dA, hA, (size_t)M * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, hSFA, (size_t)M * Ksf, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, hSFB, (size_t)N * Ksf, cudaMemcpyHostToDevice);

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
    for (int i = 0; i < 5; i++) matmul_fp4<<<grid, block, SMEM>>>(mapA, mapB, dC, N, Kf, dSFA, dSFB, Ksf);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) matmul_fp4<<<grid, block, SMEM>>>(mapA, mapB, dC, N, Kf, dSFA, dSFB, Ksf);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    double gflops = 2.0 * M * N * Kf / (ms / 1e3) / 1e9;

    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verify) {
        int wrong = 0; float maxrel = 0.f;
        for (int m = 0; m < M && wrong < 10; m++)
            for (int n = 0; n < N; n++) {
                float ref = 0.f;
                for (int k = 0; k < Kf; k++) {
                    uint8_t ba = hA[(size_t)m * Kb + k / 2], bb = hB[(size_t)n * Kb + k / 2];
                    float av = decode_fp4(k & 1 ? ba >> 4 : ba & 0xf) * ldexpf(1.f, (int)hSFA[(size_t)m * Ksf + k / 32] - 127);
                    float bv = decode_fp4(k & 1 ? bb >> 4 : bb & 0xf) * ldexpf(1.f, (int)hSFB[(size_t)n * Ksf + k / 32] - 127);
                    ref += av * bv;
                }
                float rel = fabsf(hC[(size_t)m * N + n] - ref) / (fabsf(ref) + 1.f);
                if (rel > maxrel) maxrel = rel;
                if (rel > 1e-2f) wrong++;
            }
        printf("ppsc VERIFY %dx%dx%d: %s (%d wrong, maxrel %.4f)\n", M, N, Kf, wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    } else {
        printf("matmul_fp4_ppsc (scaled): %dx%dx%d  %.3f ms  %.1f GFLOP/s\n", M, N, Kf, ms, gflops);
    }
    free(hA); free(hB); free(hSFA); free(hSFB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dC);
}

int main() {
    run(512, true);   // correctness vs host reference
    run(2048, false);
    run(4096, false);
    run(8192, false);
    return 0;
}
