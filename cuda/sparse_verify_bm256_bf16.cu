// Raw-PTX track: 2:4-sparse FP4 matmul, traffic-optimal tiling + decoupled async pipeline.
// The mem-only probe proved the 256x128 shared-B tiling cuts the memory ceiling from
// 1620k -> 2037k (20% less B traffic). But a CTA-wide __syncthreads exposed compute
// (1448k < 2037k ceiling). This version replaces the CTA barrier with a two-mbarrier
// full/empty pipeline: consumers wait `full` (data ready), compute, signal `empty`; the
// producer waits `empty` (buffer free) before refilling. No CTA-wide lockstep, so a fast
// warp runs ahead into the next stage while the producer refills => compute hides behind
// the lower traffic. All-ones => out == Klog/2.

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#define BM 128
#define BN 128
#define BKL 128
#define AROWB 32
#define BROWB 64
#define STAGES 6
#define ASZ (BM * AROWB)
#define BSZ (BN * BROWB)
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + 2 * STAGES * 8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          __nv_bfloat16 *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;

    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);
    uint64_t *empty = full + STAGES;

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

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

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
        // signal buffer s free, then producer refills it once all 256 consumers signaled
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) {
            asm volatile("{\n\t.reg .pred p;\n\tWE:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra WE;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])), "r"(par));
            issue(s, next);
        }
    }

#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2);
            int gc = block_col + b_col[n] + (lane & 3) * 2;
            *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0], acc[idx][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2], acc[idx][3]);
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

static float decode_fp4(uint8_t n) {
    int sign = (n >> 3) & 1, exp = (n >> 1) & 3, man = n & 1;
    float v = (exp == 0) ? (man ? 0.5f : 0.f) : ((float)(1 << (exp - 1)) * (1.f + 0.5f * man));
    return sign ? -v : v;
}

// Real-data correctness for the bf16-output 256x128 kernel: random pair-granular 2:4 A
// (compressed) and random B, unit ue4m3 scales. bf16 store rounds the f32 accumulation,
// so tolerance is 2^-8 (~4e-3) relative, not exact.
void run(int sz) {
    int M = sz, N = sz, Klog = sz, KAb = Klog / 4, KBb = Klog / 2;
    uint8_t *dA, *dB;
    __nv_bfloat16 *dC;
    cudaMalloc(&dA, (size_t)M * KAb);
    cudaMalloc(&dB, (size_t)N * KBb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(__nv_bfloat16));

    uint8_t *hAc = (uint8_t *)malloc((size_t)M * KAb), *hB = (uint8_t *)malloc((size_t)N * KBb);
    uint8_t *hAlog = (uint8_t *)calloc((size_t)M * Klog, 1);
    uint32_t st = 0x5A5Au;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (int i = 0; i < M; i++)
        for (int cs = 0; cs < KAb; cs++) {
            int pp = (cs / 2) * 4 + (cs % 2);
            uint8_t lo = rnd() & 0xf, hi = rnd() & 0xf;
            hAc[(size_t)i * KAb + cs] = lo | (hi << 4);
            hAlog[(size_t)i * Klog + 2 * pp + 0] = lo;
            hAlog[(size_t)i * Klog + 2 * pp + 1] = hi;
        }
    for (size_t i = 0; i < (size_t)N * KBb; i++) hB[i] = (uint8_t)rnd();
    cudaMemcpy(dA, hAc, (size_t)M * KAb, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * KBb, cudaMemcpyHostToDevice);

    alignas(64) CUtensorMap mapA, mapB;
    mk("A", &mapA, dA, KAb, M, AROWB, BM);
    mk("B", &mapB, dB, KBb, N, BROWB, BN);

    cudaFuncSetAttribute(matmul_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / BN, M / (2 * BM)), block(256);
    matmul_sp<<<grid, block, SMEM>>>(mapA, mapB, dC, M, N, Klog);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }

    __nv_bfloat16 *hC = (__nv_bfloat16 *)malloc((size_t)M * N * sizeof(__nv_bfloat16));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int wrong = 0; float maxrel = 0.f;
    for (int i = 0; i < M && wrong < 8; i++)
        for (int j = 0; j < N; j++) {
            float ref = 0.f;
            for (int k = 0; k < Klog; k++) {
                if (((k / 2) & 3) >= 2) continue;
                uint8_t bn = hB[(size_t)j * KBb + k / 2];
                float bv = decode_fp4(k & 1 ? bn >> 4 : bn & 0xf);
                ref += decode_fp4(hAlog[(size_t)i * Klog + k]) * bv;
            }
            float got = __bfloat162float(hC[(size_t)i * N + j]);
            float rel = fabsf(got - ref) / (fabsf(ref) + 1.f);
            if (rel > maxrel) maxrel = rel;
            if (rel > 5e-3f) { if (wrong < 4) printf("  [%d][%d] got %.3f ref %.3f\n", i, j, got, ref); wrong++; }
        }
    printf("sparse_verify_bm256_bf16 (256x128 bf16 out, random A/B, unit scales): %dx%dx%d  %s (%d wrong, maxrel %.5f, tol 5e-3)\n",
           M, N, Klog, wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    free(hAc); free(hB); free(hAlog); free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(512);
    run(1024);
    return 0;
}
