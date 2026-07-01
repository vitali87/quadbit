// DEPLOYABLE NVFP4 dense FP4 (e2m1 + per-16 ue4m3 scales, scale_vec::4X), PyTorch-callable.
// Same fast pingpong tiling + coalesced smem-staged scales as the MXFP4 kernel, but per-16 NVFP4
// scales (ue4m3) for higher accuracy (~0.10 vs MXFP4 0.165). STAGES=2 (per-16 scales are 2x the
// bytes -> don't fit STAGES=3 smem). SFA[step][M][8]/SFB[step][N][8] ue4m3 (8 per-16 blocks/128-step).
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define DBM 128
#define DBN 128
#define DBK 128
#define DBKH 64
#define DSTAGES 2
#define DASZ (DBM * DBKH)
#define DBSZ (DBN * DBKH)
#define DWG (DSTAGES * DASZ + DSTAGES * DBSZ)
#define SCB 1024                     // one step's scales per WG: 128 rows/cols x 8 per-16 blocks (bytes)
#define DSMEM (2 * DWG + 2 * DSTAGES * 8 + 16 + 2 * 2 * SCB + 128)

__global__ void __launch_bounds__(256)
dmatmul_nvf(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
            __nv_bfloat16 *C, int M, int N, int Kfp4, const uint8_t *SFA, const uint8_t *SFB, int Ksf16) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem + wg * DWG;
    uint8_t *b_s = a_s + DSTAGES * DASZ;
    uint64_t *full = (uint64_t *)(smem + 2 * DWG + wg * DSTAGES * 8);
    uint8_t *sca = smem + 2 * DWG + 2 * DSTAGES * 8 + 16 + wg * (2 * SCB);
    uint8_t *scb = sca + SCB;
    int sync_id = wg + 1;
    int block_row = blockIdx.y * DBM, block_col = blockIdx.x * (2 * DBN) + wg * DBN;
    int ksteps = Kfp4 / DBK;
    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;
    int sa_row = (lane >> 2) + 8 * (lane & 1), sa_ok = (lane & 3) < 2;
    int sb_col = lane >> 2, sb_ok = (lane & 3) == 0;
    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint16_t z = 0;
    if (wtid == 0) {
#pragma unroll
        for (int s = 0; s < DSTAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
    auto issue = [&](int s, int kstep) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(DASZ + DBSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&a_s[s * DASZ])), "l"(&mapA), "r"(kstep * DBKH), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * DBSZ])), "l"(&mapB), "r"(kstep * DBKH), "r"(block_col), "r"(bar));
    };
    if (wtid == 0)
#pragma unroll
        for (int s = 0; s < DSTAGES; s++)
            if (s < ksteps) issue(s, s);
    for (int step = 0; step < ksteps; step++) {
        int s = step % DSTAGES;
        uint32_t par = (step / DSTAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        {   // stage step's per-16 scales (step-major -> contiguous 1024B; coalesced uint32 loads)
            const uint32_t *srcA = (const uint32_t *)(SFA + (size_t)step * M * 8 + block_row * 8);
            const uint32_t *srcB = (const uint32_t *)(SFB + (size_t)step * N * 8 + block_col * 8);
            uint32_t *dstA = (uint32_t *)sca, *dstB = (uint32_t *)scb;
            dstA[wtid] = srcA[wtid]; dstA[wtid + 128] = srcA[wtid + 128];   // 256 uint32 = 1024B
            dstB[wtid] = srcB[wtid]; dstB[wtid + 128] = srcB[wtid + 128];
        }
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
        int aoff = s * DASZ, boff = s * DBSZ;
#pragma unroll
        for (int ks = 0; ks < DBK / 64; ks++) {
            int kb = ks * 32;
            uint32_t af[4][4], bf[8][2], saR[4], sbR[8];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * DBKH + kb + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
                saR[mt] = sa_ok ? *reinterpret_cast<const uint32_t *>(&sca[(a_rowt[mt] + sa_row) * 8 + ks * 4]) : 0;
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * DBKH + kb + bkblk * 16; bo ^= ((bo >> 7) & 3) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]) : "r"(bd));
                sbR[n] = sb_ok ? *reinterpret_cast<const uint32_t *>(&scb[(b_col[n] + sb_col) * 8 + ks * 4]) : 0;
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]), "r"(bf[n][0]), "r"(bf[n][1]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(saR[mt]), "h"(z), "h"(z), "r"(sbR[n]), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
        int next = step + DSTAGES;
        if (wtid == 0 && next < ksteps) issue(s, next);
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + b_col[n] + (lane & 3) * 2;
            *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0], acc[idx][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2], acc[idx][3]);
        }
}

__device__ __forceinline__ uint8_t enc_ue4m3(float s) {
    if (!(s > 0.f)) return 0;
    if (s >= 448.f) return 0x7e;
    int e; float m = frexpf(s, &e);
    float mm = 2.f * m; int biased = (e - 1) + 7;
    if (biased < 1) return 1;
    int mant = __float2int_rn((mm - 1.f) * 8.f);
    if (mant == 8) { mant = 0; biased++; }
    if (biased > 15) return 0x7e;
    return (uint8_t)((biased << 3) | mant);
}
__device__ __forceinline__ float dec_ue4m3(uint8_t n) {
    int e = (n >> 3) & 0xf, m = n & 7;
    return e == 0 ? (float)m * 0.001953125f : (1.f + m / 8.f) * exp2f((float)(e - 7));
}
__device__ __forceinline__ uint8_t q_fp4(float q) {
    float a = fabsf(q);
    int idx = a < .25f ? 0 : a < .75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3
            : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.f ? 6 : 7;
    return (uint8_t)(idx | (q < 0.f ? 8 : 0));
}
// per-16 NVFP4 activation quantizer -> Bbytes[batch,in/2] + SFB[step][batch][8] ue4m3 (step-major)
__global__ void quant_act_nv(const int4 *x4, uint32_t *Bwords, uint8_t *SFB, int batch, int in_f) {
    int b16 = in_f / 16;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * b16) return;
    int n = t / b16, blk = t % b16, step = blk / 8, kb = blk % 8;   // 8 per-16 blocks per 128-step
    long base = (long)n * (in_f / 8) + blk * 2;                     // 16 fp4 = 2 int4 (8 bf16 each)
    float2 v[8]; float amax = 0.f;
#pragma unroll
    for (int q = 0; q < 2; q++) {
        int4 p = x4[base + q];
        const __nv_bfloat162 *bp = (const __nv_bfloat162 *)&p;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[q * 4 + j] = __bfloat1622float2(bp[j]);
            amax = fmaxf(amax, fmaxf(fabsf(v[q * 4 + j].x), fabsf(v[q * 4 + j].y)));
        }
    }
    uint8_t code = enc_ue4m3(amax / 6.f);
    SFB[((long)step * batch + n) * 8 + kb] = code;
    float inv = 1.f / dec_ue4m3(code);
    uint32_t w2[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t byte = q_fp4(v[i].x * inv) | (q_fp4(v[i].y * inv) << 4);
        w2[i >> 2] |= byte << ((i & 3) * 8);
    }
    *reinterpret_cast<uint2 *>(Bwords + (long)n * (in_f / 8) + blk * 2) = make_uint2(w2[0], w2[1]);
}
extern "C" void quantize_act_nvfp4b(const void *x, void *Bbytes, void *SFB, int batch, int in_f) {
    int total = batch * (in_f / 16), tpb = 256;
    quant_act_nv<<<(total + tpb - 1) / tpb, tpb>>>((const int4 *)x, (uint32_t *)Bbytes, (uint8_t *)SFB, batch, in_f);
}

// ---- NVFP4 fused ops (per-16 ue4m3, step-major SFB[step][batch][8]) for the NVFP4 dense kernel.
__device__ __forceinline__ void nv_pack16(const float *val, uint32_t *Bwords, uint8_t *SFB,
                                           int batch, int in_f, int n, int blk) {
    int step = blk / 8, kb = blk % 8;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(val[i]));
    uint8_t code = enc_ue4m3(amax / 6.f);
    SFB[((long)step * batch + n) * 8 + kb] = code;
    float inv = 1.f / dec_ue4m3(code);
    uint32_t w2[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t byte = q_fp4(val[2 * i] * inv) | (q_fp4(val[2 * i + 1] * inv) << 4);
        w2[i >> 2] |= byte << ((i & 3) * 8);
    }
    *reinterpret_cast<uint2 *>(Bwords + (long)n * (in_f / 8) + blk * 2) = make_uint2(w2[0], w2[1]);
}
__global__ void rmsnorm_nv_k(const __nv_bfloat16 *x, const __nv_bfloat16 *res, const __nv_bfloat16 *wt,
                             __nv_bfloat16 *hout, uint32_t *Bwords, uint8_t *SFB, int batch, int hidden, float eps) {
    int b = blockIdx.x;
    extern __shared__ float sm[];
    const __nv_bfloat16 *xr = x + (long)b * hidden;
    float ls = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        if (res) { v += __bfloat162float(res[(long)b * hidden + i]); hout[(long)b * hidden + i] = __float2bfloat16_rn(v); }
        sm[i] = v; ls += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ls += __shfl_down_sync(0xffffffffu, ls, o);
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ls;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0; int nw = blockDim.x >> 5; for (int i = 0; i < nw; i++) t += red[i]; red[0] = rsqrtf(t / hidden + eps); }
    __syncthreads();
    float rms = red[0];
    for (int blk = threadIdx.x; blk < hidden / 16; blk += blockDim.x) {
        float val[16];
#pragma unroll
        for (int i = 0; i < 16; i++) val[i] = sm[blk * 16 + i] * rms * __bfloat162float(wt[blk * 16 + i]);
        nv_pack16(val, Bwords, SFB, batch, hidden, b, blk);
    }
}
extern "C" void rmsnorm_nvfp4(const void *x, const void *res, const void *wt, void *hout,
                              void *Bbytes, void *SFB, int batch, int hidden, float eps) {
    int smem = hidden * (int)sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(rmsnorm_nv_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    rmsnorm_nv_k<<<batch, 256, smem>>>((const __nv_bfloat16 *)x, (const __nv_bfloat16 *)res,
                                       (const __nv_bfloat16 *)wt, (__nv_bfloat16 *)hout,
                                       (uint32_t *)Bbytes, (uint8_t *)SFB, batch, hidden, eps);
}
__global__ void swiglu_nv_k(const __nv_bfloat16 *g, const __nv_bfloat16 *u, uint32_t *Hwords,
                            uint8_t *SFB, int batch, int hidden) {
    int hb = hidden / 16;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * hb) return;
    int blk = (int)(t / batch), b = (int)(t % batch);
    long base = (long)(blk * 16) * batch + b;
    float val[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        float gv = __bfloat162float(g[base + (long)i * batch]), uv = __bfloat162float(u[base + (long)i * batch]);
        val[i] = (gv / (1.f + __expf(-gv))) * uv;
    }
    nv_pack16(val, Hwords, SFB, batch, hidden, b, blk);
}
extern "C" void swiglu_nvfp4(const void *g, const void *u, void *Hbytes, void *SFB, int batch, int hidden) {
    int total = batch * (hidden / 16), tpb = 256;
    swiglu_nv_k<<<(total + tpb - 1) / tpb, tpb>>>((const __nv_bfloat16 *)g, (const __nv_bfloat16 *)u,
                                                  (uint32_t *)Hbytes, (uint8_t *)SFB, batch, hidden);
}

static void mkmap(CUtensorMap *m, uint8_t *p, int Kb, int rows, int boxrows) {
    uint64_t gd[2] = {(uint64_t)Kb, (uint64_t)rows}; uint64_t gs[1] = {(uint64_t)Kb};
    uint32_t bd[2] = {(uint32_t)DBKH, (uint32_t)boxrows}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map fail %s\n", s); }
}

extern "C" int dense_nvfp4_mm(const void *A, const void *B, const void *SFA, const void *SFB,
                              void *C, int M, int N, int K) {
    int Kb = K / 2, Ksf16 = K / 16;
    alignas(64) CUtensorMap mapA, mapB;
    mkmap(&mapA, (uint8_t *)A, Kb, M, DBM);
    mkmap(&mapB, (uint8_t *)B, Kb, N, DBN);
    cudaFuncSetAttribute(dmatmul_nvf, cudaFuncAttributeMaxDynamicSharedMemorySize, DSMEM);
    dim3 grid(N / (2 * DBN), M / DBM), block(256);
    dmatmul_nvf<<<grid, block, DSMEM>>>(mapA, mapB, (__nv_bfloat16 *)C, M, N, K,
                                        (const uint8_t *)SFA, (const uint8_t *)SFB, Ksf16);
    return (int)cudaDeviceSynchronize();
}
