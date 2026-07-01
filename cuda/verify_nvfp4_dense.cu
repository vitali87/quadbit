// Probe: DENSE FP4 m16n8k64 with NVFP4 scales (per-16 blocks, ue4m3, scale_vec::4X). Derives the
// dense 4X scale lane layout by verifying against a host reference. Hypothesis: the A/B row->lane
// mapping is the SAME as the ue8m0 2X case (it's fixed by m16n8k64); scale_vec::4X just widens the
// scale reg to 4 bytes (4 per-16 scales for the k64). SFA[row][K/16]/SFB[col][K/16] ue4m3, random
// data + random ue4m3 scales near 1.0. PASS => the layout is confirmed and NVFP4 dense is buildable.
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
           float *C, int N, int Kfp4, const uint8_t *SFA, const uint8_t *SFB, int Ksf16) {
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
    uint16_t z = 0;
    int sa_row = (lane >> 2) + 8 * (lane & 1), sa_ok = (lane & 3) < 2;
    int sb_col = lane >> 2, sb_ok = (lane & 3) == 0;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();
    auto issue = [&](int s, int kstep) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&a_s[s * ASZ])), "l"(&mapA), "r"(kstep * BKH), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(kstep * BKH), "r"(block_col), "r"(bar));
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < ksteps) issue(s, s);
    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WAIT;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int ks = 0; ks < BK / 64; ks++) {
            int kb = ks * 32;
            int k16 = step * (BK / 16) + ks * 4;       // first of 4 per-16 blocks for this k64
            uint32_t af[2][4], bf[8][2], saR[2], sbR[8];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                int arw = m == 0 ? a_row0 : a_row1;
                int ao = (arw + arow) * BKH + kb + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3]) : "r"(ad));
                saR[m] = 0;
                if (sa_ok) saR[m] = *reinterpret_cast<const uint32_t *>(&SFA[(size_t)(block_row + arw + sa_row) * Ksf16 + k16]);
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BKH + kb + bkblk * 16; bo ^= ((bo >> 7) & 3) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]) : "r"(bd));
                sbR[n] = 0;
                if (sb_ok) sbR[n] = *reinterpret_cast<const uint32_t *>(&SFB[(size_t)(block_col + b_col[n] + sb_col) * Ksf16 + k16]);
            }
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = m * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]), "r"(bf[n][0]), "r"(bf[n][1]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(saR[m]), "h"(z), "h"(z), "r"(sbR[n]), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        __syncthreads();
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) issue(s, next);
    }
#pragma unroll
    for (int m = 0; m < 2; m++) {
        int arw = m == 0 ? a_row0 : a_row1;
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = m * 8 + n;
            int gr = block_row + arw + (lane >> 2), gc = block_col + b_col[n] + (lane & 3) * 2;
            C[gr * N + gc] = acc[idx][0]; C[gr * N + gc + 1] = acc[idx][1];
            C[(gr + 8) * N + gc] = acc[idx][2]; C[(gr + 8) * N + gc + 1] = acc[idx][3];
        }
    }
}

static float decode_fp4(uint8_t n) {
    int sign = (n >> 3) & 1, exp = (n >> 1) & 3, man = n & 1;
    float v = (exp == 0) ? (man ? 0.5f : 0.f) : ((float)(1 << (exp - 1)) * (1.f + 0.5f * man));
    return sign ? -v : v;
}
static float dec_ue4m3(uint8_t n) {
    int e = (n >> 3) & 0xf, m = n & 7;
    return e == 0 ? (float)m * 0.001953125f : (1.f + m / 8.f) * ldexpf(1.f, e - 7);
}

void run(int sz) {
    int M = sz, N = sz, Kf = sz, Kb = Kf / 2, Ksf16 = Kf / 16;
    uint8_t *dA, *dB, *dSFA, *dSFB; float *dC;
    cudaMalloc(&dA, (size_t)M * Kb); cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dSFA, (size_t)M * Ksf16); cudaMalloc(&dSFB, (size_t)N * Ksf16);
    cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    uint8_t *hA = (uint8_t *)malloc((size_t)M * Kb), *hB = (uint8_t *)malloc((size_t)N * Kb);
    uint8_t *hSFA = (uint8_t *)malloc((size_t)M * Ksf16), *hSFB = (uint8_t *)malloc((size_t)N * Ksf16);
    uint32_t st = 0xC0FFEEu;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (size_t i = 0; i < (size_t)M * Kb; i++) hA[i] = (uint8_t)rnd();
    for (size_t i = 0; i < (size_t)N * Kb; i++) hB[i] = (uint8_t)rnd();
    for (size_t i = 0; i < (size_t)M * Ksf16; i++) hSFA[i] = 40 + (rnd() % 32);  // WIDE ue4m3 range ~2^-4..2^2
    for (size_t i = 0; i < (size_t)N * Ksf16; i++) hSFB[i] = 40 + (rnd() % 32);
    cudaMemcpy(dA, hA, (size_t)M * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)N * Kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, hSFA, (size_t)M * Ksf16, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, hSFB, (size_t)N * Ksf16, cudaMemcpyHostToDevice);
    auto build = [](uint8_t *p, int rows, int Kb, int boxrows) {
        CUtensorMap m; uint64_t gd[2] = {(uint64_t)Kb, (uint64_t)rows}; uint64_t gs[1] = {(uint64_t)Kb};
        uint32_t bd[2] = {(uint32_t)BKH, (uint32_t)boxrows}, es[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map failed: %s\n", s); }
        return m;
    };
    CUtensorMap mapA = build(dA, M, Kb, BM), mapB = build(dB, N, Kb, BN);
    dim3 grid(N / BN, M / BM), block(128);
    matmul_fp4<<<grid, block>>>(mapA, mapB, dC, N, Kf, dSFA, dSFB, Ksf16);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return; }
    float *hC = (float *)malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0; float maxrel = 0.f;
    for (int m = 0; m < M && wrong < 10; m++)
        for (int n = 0; n < N; n++) {
            float ref = 0.f;
            for (int k = 0; k < Kf; k++) {
                uint8_t ba = hA[(size_t)m * Kb + k / 2], bb = hB[(size_t)n * Kb + k / 2];
                float av = decode_fp4(k & 1 ? ba >> 4 : ba & 0xf), bv = decode_fp4(k & 1 ? bb >> 4 : bb & 0xf);
                float sa = dec_ue4m3(hSFA[(size_t)m * Ksf16 + k / 16]), sb = dec_ue4m3(hSFB[(size_t)n * Ksf16 + k / 16]);
                ref += (av * sa) * (bv * sb);
            }
            float got = hC[(size_t)m * N + n], rel = fabsf(got - ref) / (fabsf(ref) + 1.f);
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-2f) { if (wrong < 5) printf("  mismatch [%d][%d]: got %.3f ref %.3f\n", m, n, got, ref); wrong++; }
        }
    printf("verify_nvfp4_dense (per-16 ue4m3 scale_vec::4X): %dx%dx%d  %s (%d wrong, maxrel %.4f)\n",
           M, N, Kf, wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    free(hA); free(hB); free(hSFA); free(hSFB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dC);
}

int main() { run(2048); return 0; }
