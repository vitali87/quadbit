// Raw-PTX track: scaled pingpong with TMA-STAGED scale factors. The 955k cooperative
// version loads SF with uncoalesced global reads + an extra bar.sync. Here the SF
// tensors are stored TRANSPOSED ([K/32][M], kblock-major) so a TMA box {128 rows, 4
// kblocks} has a 16B-aligned inner dim (128 bytes) and pulls exactly one step's scales
// per TMA, landing via an mbarrier (no uncoalesced load, no extra warpgroup barrier).
// Single SF buffer per WG (fits the 3-stage data smem budget); verified vs host ref.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 128
#define BN 128
#define BK 128
#define BKH 64
#define STAGES 3
#define ASZ (BM * BKH)
#define BSZ (BN * BKH)
#define WGBYTES (STAGES * ASZ + STAGES * BSZ)
#define SFTILE 512                                       // 4 kblocks * 128 rows bytes
#define SMEM (2 * WGBYTES + 2 * (2 * SFTILE) + 16 + 2 * STAGES * 8 + 128)

__global__ void __launch_bounds__(256)
matmul_fp4(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
           const __grid_constant__ CUtensorMap mapSFA, const __grid_constant__ CUtensorMap mapSFB,
           float *C, int N, int Kfp4) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;

    uint8_t *a_s = smem + wg * WGBYTES;
    uint8_t *b_s = a_s + STAGES * ASZ;
    // SF tiles right after the data so they're 128-byte aligned (TMA smem dst needs it);
    // mbarriers placed after, where alignment doesn't matter.
    uint8_t *sfa_s = smem + 2 * WGBYTES + wg * (2 * SFTILE);  // [4 kb][128 row], 128-aligned
    uint8_t *sfb_s = sfa_s + SFTILE;
    uint64_t *sf_full = (uint64_t *)(smem + 2 * WGBYTES + 2 * (2 * SFTILE) + wg * 8);
    uint64_t *full = (uint64_t *)(smem + 2 * WGBYTES + 2 * (2 * SFTILE) + 16 + wg * STAGES * 8);
    int sync_id = wg + 1;

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
    int sa_lrow = (lane >> 2) + 8 * (lane & 1);
    int sb_lcol = lane >> 2;

    if (wtid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(sf_full)));
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
    auto issueSF = [&](int step) {  // load this step's 4 kblocks x 128 rows for A and B
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(sf_full);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(2 * SFTILE)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(sfa_s)),
                     "l"(&mapSFA), "r"(block_row), "r"(step * 4), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(sfb_s)),
                     "l"(&mapSFB), "r"(block_col), "r"(step * 4), "r"(bar));
    };

    if (wtid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < ksteps) issue(s, s);
        issueSF(0);
    }

    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\n\tWAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra WAIT;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        // wait this step's SF (single buffer, parity flips each step)
        asm volatile("{\n\t.reg .pred p;\n\tWS:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra WS;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(sf_full)), "r"((uint32_t)(step & 1)));

        // read this lane's scales from the transposed SF smem tile (sf[kb*128 + row])
        uint32_t saC[4], sbC[8];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int r = a_rowt[mt] + sa_lrow;
            saC[mt] = sfa_s[0 * 128 + r] | (sfa_s[1 * 128 + r] << 8) | (sfa_s[2 * 128 + r] << 16) | (sfa_s[3 * 128 + r] << 24);
        }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int c = b_col[n] + sb_lcol;
            sbC[n] = sfb_s[0 * 128 + c] | (sfb_s[1 * 128 + c] << 8) | (sfb_s[2 * 128 + c] << 16) | (sfb_s[3 * 128 + c] << 24);
        }

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
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
        int next = step + STAGES;
        if (wtid == 0) {
            if (next < ksteps) issue(s, next);
            if (step + 1 < ksteps) issueSF(step + 1);  // single buffer: reuse after the bar.sync above
        }
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

static void mk(const char *tag, CUtensorMap *m, uint8_t *p, int inner, int outer, int boxin, int boxout, int sw) {
    uint64_t gdim[2] = {(uint64_t)inner, (uint64_t)outer};
    uint64_t gstride[1] = {(uint64_t)inner};
    uint32_t bdim[2] = {(uint32_t)boxin, (uint32_t)boxout};
    uint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gdim, gstride, bdim, estride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, (CUtensorMapSwizzle)sw,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s);
        printf("map[%s] failed: %s (gdim %d,%d gstride %d box %d,%d)\n", tag, s, inner, outer, inner, boxin, boxout); }
}

void run(int sz, bool verify) {
    int M = sz, N = sz, Kf = sz, Kb = Kf / 2, Ksf = Kf / 32;
    uint8_t *dA, *dB, *dSFA, *dSFB;
    float *dC;
    cudaMalloc(&dA, (size_t)M * Kb);
    cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dSFA, (size_t)M * Ksf);  // transposed [Ksf][M]
    cudaMalloc(&dSFB, (size_t)N * Ksf);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));

    uint8_t *hA = (uint8_t *)malloc((size_t)M * Kb), *hB = (uint8_t *)malloc((size_t)N * Kb);
    uint8_t *hSFA = (uint8_t *)malloc((size_t)M * Ksf), *hSFB = (uint8_t *)malloc((size_t)N * Ksf);
    uint8_t *hSFAt = (uint8_t *)malloc((size_t)M * Ksf), *hSFBt = (uint8_t *)malloc((size_t)N * Ksf);
    uint32_t st = 0xD00Du;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (size_t i = 0; i < (size_t)M * Kb; i++) hA[i] = verify ? (uint8_t)rnd() : 0x22;
    for (size_t i = 0; i < (size_t)N * Kb; i++) hB[i] = verify ? (uint8_t)rnd() : 0x22;
    for (size_t i = 0; i < (size_t)M * Ksf; i++) hSFA[i] = verify ? (125 + rnd() % 5) : 127;
    for (size_t i = 0; i < (size_t)N * Ksf; i++) hSFB[i] = verify ? (125 + rnd() % 5) : 127;
    // transpose to [Ksf][M] / [Ksf][N]
    for (int r = 0; r < M; r++) for (int k = 0; k < Ksf; k++) hSFAt[(size_t)k * M + r] = hSFA[(size_t)r * Ksf + k];
    for (int c = 0; c < N; c++) for (int k = 0; k < Ksf; k++) hSFBt[(size_t)k * N + c] = hSFB[(size_t)c * Ksf + k];
    cudaMemcpy(dA, hA, (size_t)M * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, hSFAt, (size_t)M * Ksf, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, hSFBt, (size_t)N * Ksf, cudaMemcpyHostToDevice);

    alignas(64) CUtensorMap mapA, mapB, mapSFA, mapSFB;
    mk("A", &mapA, dA, Kb, M, BKH, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    mk("B", &mapB, dB, Kb, N, BKH, BN, CU_TENSOR_MAP_SWIZZLE_64B);
    mk("SFA", &mapSFA, dSFA, M, Ksf, BM, 4, CU_TENSOR_MAP_SWIZZLE_NONE);  // [Ksf][M], box {128 rows, 4 kb}
    mk("SFB", &mapSFB, dSFB, N, Ksf, BN, 4, CU_TENSOR_MAP_SWIZZLE_NONE);

    cudaFuncSetAttribute(matmul_fp4, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / (2 * BN), M / BM), block(256);
    for (int i = 0; i < 5; i++) matmul_fp4<<<grid, block, SMEM>>>(mapA, mapB, mapSFA, mapSFB, dC, N, Kf);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    int iters = 20;
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) matmul_fp4<<<grid, block, SMEM>>>(mapA, mapB, mapSFA, mapSFB, dC, N, Kf);
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
        printf("ppsc_tma VERIFY %dx%dx%d: %s (%d wrong, maxrel %.4f)\n", M, N, Kf, wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    } else {
        printf("matmul_fp4_ppsc_tma (TMA scales): %dx%dx%d  %.3f ms  %.1f GFLOP/s\n", M, N, Kf, ms, gflops);
    }
    free(hA); free(hB); free(hSFA); free(hSFB); free(hSFAt); free(hSFBt); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dC);
}

int main() {
    run(512, true);
    run(2048, false);
    run(4096, false);
    run(8192, false);
    return 0;
}
