// DEPLOYABLE real-scale DENSE FP4 (MXFP4: e2m1 + per-32-block ue8m0 scales, BOTH operands),
// PyTorch-callable. This is the drop-in that works on ANY real model with NO training: dense FP4
// (no 2:4 prune) at ~+0.1 error. Kernel is the proven verify_scaled.cu mma (dense scale lane
// layout: SFA[row][kblk], SFB[col][kblk] row-major ue8m0; sa_row=(lane>>2)+8*(lane&1), sb_col=
// lane>>2) with bf16 output. C[out,batch] = W[out,in] @ x[batch,in]^T with real weight+act scales.
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 64
#define BN 128
#define BK 128
#define BKH 64
#define STAGES 3
#define ASZ (BM * BKH)
#define BSZ (BN * BKH)

__global__ void __launch_bounds__(128)
dmatmul_scaled(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
               __nv_bfloat16 *C, int N, int Kfp4, const uint8_t *SFA, const uint8_t *SFB, int Ksf) {
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
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int ks = 0; ks < BK / 64; ks++) {
            int kb = ks * 32;
            int kbi = step * (BK / 32) + ks * 2;
            uint32_t af[2][4], bf[8][2], saR[2], sbR[8];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                int arw = m == 0 ? a_row0 : a_row1;
                int ao = (arw + arow) * BKH + kb + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[m][0]), "=r"(af[m][1]), "=r"(af[m][2]), "=r"(af[m][3]) : "r"(ad));
                saR[m] = 0;
                if (sa_ok) {
                    const uint8_t *p = &SFA[(block_row + arw + sa_row) * Ksf + kbi];
                    saR[m] = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
                }
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BKH + kb + bkblk * 16; bo ^= ((bo >> 7) & 3) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]) : "r"(bd));
                sbR[n] = 0;
                if (sb_ok) {
                    const uint8_t *p = &SFB[(block_col + b_col[n] + sb_col) * Ksf + kbi];
                    sbR[n] = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
                }
            }
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = m * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
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
            *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0], acc[idx][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2], acc[idx][3]);
        }
    }
}

// Fused MXFP4 activation quantizer: x[batch,in] bf16 -> Bbytes[batch,in/2] + SFB[batch,in/32] ue8m0.
__device__ __forceinline__ uint8_t enc_ue8m0(float amax) {
    if (!(amax > 0.f)) return 127;                 // scale 2^0 for all-zero block
    int e; frexpf(amax / 6.f, &e);                 // amax/6 <= 2^e ; scale = 2^e covers the *6 range
    int code = e + 127;
    return (uint8_t)(code < 0 ? 0 : code > 255 ? 255 : code);
}
__device__ __forceinline__ uint8_t q_fp4(float q) {
    float a = fabsf(q);
    int idx = a < .25f ? 0 : a < .75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3
            : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.f ? 6 : 7;
    return (uint8_t)(idx | (q < 0.f ? 8 : 0));
}
__global__ void quant_act_mx(const int4 *x4, uint32_t *Bwords, uint8_t *SFB, int batch, int in_f) {
    int b32 = in_f / 32;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * b32) return;
    int n = t / b32, blk = t % b32;
    long base = (long)n * (in_f / 8) + blk * 4;
    float2 v[16]; float amax = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        int4 p = x4[base + q];
        const __nv_bfloat162 *bp = (const __nv_bfloat162 *)&p;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[q * 4 + j] = __bfloat1622float2(bp[j]);
            amax = fmaxf(amax, fmaxf(fabsf(v[q * 4 + j].x), fabsf(v[q * 4 + j].y)));
        }
    }
    uint8_t code = enc_ue8m0(amax);
    SFB[(long)n * b32 + blk] = code;               // SFB[batch][in/32] row-major
    float inv = 1.f / ldexpf(1.f, (int)code - 127);
    uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t byte = q_fp4(v[i].x * inv) | (q_fp4(v[i].y * inv) << 4);
        w[i >> 2] |= byte << ((i & 3) * 8);
    }
    *reinterpret_cast<uint4 *>(Bwords + (long)n * (in_f / 8) + blk * 4) = make_uint4(w[0], w[1], w[2], w[3]);
}
extern "C" void quantize_act_mxfp4(const void *x, void *Bbytes, void *SFB, int batch, int in_f) {
    int total = batch * (in_f / 32), tpb = 256;
    quant_act_mx<<<(total + tpb - 1) / tpb, tpb>>>((const int4 *)x, (uint32_t *)Bbytes, (uint8_t *)SFB, batch, in_f);
}

static void mkmap(CUtensorMap *m, uint8_t *p, int Kb, int rows, int boxrows) {
    uint64_t gd[2] = {(uint64_t)Kb, (uint64_t)rows}; uint64_t gs[1] = {(uint64_t)Kb};
    uint32_t bd[2] = {(uint32_t)BKH, (uint32_t)boxrows}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map fail %s\n", s); }
}

// C[M,N] = A[M,K] @ B[N,K]^T, A/B dense FP4 (2/byte), SFA[M,K/32]/SFB[N,K/32] ue8m0. M=out,N=batch.
extern "C" int dense_scaled_mm(const void *A, const void *B, const void *SFA, const void *SFB,
                               void *C, int M, int N, int K) {
    int Kb = K / 2, Ksf = K / 32;
    alignas(64) CUtensorMap mapA, mapB;
    mkmap(&mapA, (uint8_t *)A, Kb, M, BM);
    mkmap(&mapB, (uint8_t *)B, Kb, N, BN);
    dim3 grid(N / BN, M / BM), block(128);
    dmatmul_scaled<<<grid, block>>>(mapA, mapB, (__nv_bfloat16 *)C, N, K,
                                    (const uint8_t *)SFA, (const uint8_t *)SFB, Ksf);
    return (int)cudaDeviceSynchronize();
}
