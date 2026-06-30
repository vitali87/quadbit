// Raw-PTX track: FULLY DEPLOYABLE 2:4-sparse FP4 matmul, PyTorch-callable.
// Wide-TMA + swizzle load path (A 64B-swz / B 128B-swz, WK=2 k128-slices per TMA) that
// broke the false roofline, plus arbitrary per-group 2:4 metadata + real ue4m3 scales.
// This is the deployable WIDE-SWIZZLE winner (2116k), not the old narrow kernel (1468k).
//   scaleA[row r][kb]->lane (r&7)*4+(r>>3) byte kb ; scaleB[col c][kb]->lane c*4 byte kb
//   metadata: lane L of m-tile -> mma-row (L&1)*8+(L>>2), half H=(L>>1)&1; e = 8 nibbles
//     for groups [H*8..H*8+8); nibble = idx0|(idx1<<2).
// Tensors STEP-MAJOR: scaleA [ksteps][M][4], scaleB [ksteps][N][4], meta [ksteps][M][2] u32.

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 128
#define BN 128
#define WK 2
#define AROWB 32
#define BROWB 64
#define AW (AROWB * WK)
#define BW_ (BROWB * WK)
#define STAGES 2
#define ASZ (BM * AW)
#define BSZ (BN * BW_)
#define SCA 1024
#define SCB 512
#define MET 2048
#define SMEM (2*STAGES*ASZ + STAGES*BSZ + STAGES*WK*SCA + STAGES*WK*SCB + STAGES*WK*MET + 2*STAGES*8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          const uint8_t *scaleA, const uint8_t *scaleB, const uint32_t *meta,
          __nv_bfloat16 *C, int M, int N, int Klog) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint8_t *scA_sm = b_s + STAGES * BSZ;
    uint8_t *scB_sm = scA_sm + STAGES * WK * SCA;
    uint8_t *met_sm = scB_sm + STAGES * WK * SCB;
    uint64_t *full = (uint64_t *)(met_sm + STAGES * WK * MET);
    uint64_t *empty = full + STAGES;

    int block_row = blockIdx.y * (2 * BM) + wg * BM;
    int a_load_row = blockIdx.y * (2 * BM), block_col = blockIdx.x * BN;
    int chunks = Klog / (128 * WK);

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;
    int ra_local = (lane & 3) * 8 + (lane >> 2), cb_local = lane >> 2;
    bool a_valid = ra_local < 16, b_valid = (lane & 3) == 0;
    int a_sidx[4], b_sidx[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_sidx[mt] = wg * 128 + a_rowt[mt] + ra_local;
#pragma unroll
    for (int n = 0; n < 8; n++) b_sidx[n] = b_col[n] + cb_local;
    int mma_row = (lane & 1) * 8 + (lane >> 2), Hh = (lane >> 1) & 1;
    int m_sidx[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) m_sidx[mt] = (wg * 128 + a_rowt[mt] + mma_row) * 2 + Hh;

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
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

    auto issue = [&](int s, int chunk) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                     "r"((uint32_t)(2 * ASZ + BSZ + WK * SCA + WK * SCB + WK * MET)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(chunk * BW_), "r"(block_col), "r"(bar));
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
            int step = chunk * WK + sub;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&scA_sm[(s * WK + sub) * SCA])), "l"(scaleA + (size_t)(step * M + a_load_row) * 4), "r"((uint32_t)SCA), "r"(bar));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&scB_sm[(s * WK + sub) * SCB])), "l"(scaleB + (size_t)(step * N + block_col) * 4), "r"((uint32_t)SCB), "r"(bar));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&met_sm[(s * WK + sub) * MET])), "l"((const uint8_t *)meta + (size_t)(step * M + a_load_row) * 8), "r"((uint32_t)MET), "r"(bar));
        }
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < chunks) issue(s, s);

    for (int chunk = 0; chunk < chunks; chunk++) {
        int s = chunk % STAGES; uint32_t par = (chunk / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
            const uint32_t *scA = (const uint32_t *)(scA_sm + (s * WK + sub) * SCA);
            const uint32_t *scB = (const uint32_t *)(scB_sm + (s * WK + sub) * SCB);
            const uint32_t *mtA = (const uint32_t *)(met_sm + (s * WK + sub) * MET);
            uint32_t sav[4], sbv[8], ev[4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) { sav[mt] = a_valid ? scA[a_sidx[mt]] : 0x38383838u; ev[mt] = mtA[m_sidx[mt]]; }
#pragma unroll
            for (int n = 0; n < 8; n++) sbv[n] = b_valid ? scB[b_sidx[n]] : 0x38383838u;
            uint32_t af[4][4], bf[8][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * AW + sub * AROWB + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BW_ + sub * BROWB + bsub * 16; bo ^= ((bo >> 7) & 7) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                          "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(ev[mt]), "r"(sav[mt]), "h"(z), "h"(z), "r"(sbv[n]), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int next = chunk + STAGES;
        if (tid == 0 && next < chunks) {
            asm volatile("{\n\t.reg .pred p;\nWE:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WE;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])), "r"(par));
            issue(s, next);
        }
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

static void mk(CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo, CUtensorMapSwizzle sw) {
    uint64_t gd[2] = {(uint64_t)inner, (uint64_t)outer}; uint64_t gs[1] = {(uint64_t)inner};
    uint32_t bd[2] = {(uint32_t)bi, (uint32_t)bo}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map fail %s\n", s); }
}

// ---- Fused NVFP4 activation quantizer: x[batch,in] f32 -> Bbytes[batch,in/2] (dense FP4,
// 2/byte, lo=even k) + scaleB[ksteps][batch][4] ue4m3 (one scale per 32-elem block). One
// memory pass; replaces the dozen eager-torch ops (which dominate the QuadbitLinear forward).
__device__ __forceinline__ uint8_t enc_ue4m3(float s) {
    if (!(s > 0.f)) return 0;
    if (s >= 480.f) return 0x7f;                       // (1+7/8)*2^8 = e4m3 max
    int e; float m = frexpf(s, &e);                    // s = m*2^e, m in [0.5,1)
    float mm = 2.f * m; int biased = (e - 1) + 7;      // s = mm*2^(e-1), mm in [1,2)
    if (biased < 1) return 1;                          // clamp tiny scales to min normal
    int mant = __float2int_rn((mm - 1.f) * 8.f);       // 0..8
    if (mant == 8) { mant = 0; biased++; }
    if (biased > 15) return 0x7f;
    return (uint8_t)((biased << 3) | mant);
}
__device__ __forceinline__ float dec_ue4m3(uint8_t n) {
    int e = (n >> 3) & 0xf, m = n & 7;
    return e == 0 ? (float)m * 0.001953125f : (1.f + m / 8.f) * exp2f((float)(e - 7));
}
__device__ __forceinline__ uint8_t q_fp4(float q) {   // nearest e2m1 code (signed)
    float a = fabsf(q);
    int idx = a < .25f ? 0 : a < .75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3
            : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.f ? 6 : 7;
    return (uint8_t)(idx | (q < 0.f ? 8 : 0));
}
__global__ void quant_act(const __nv_bfloat162 *x, uint8_t *Bbytes, uint8_t *scaleB, int batch, int in_f) {
    int b32 = in_f / 32;                               // 32-elem blocks per row
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * b32) return;
    int n = t / b32, blk = t % b32, step = blk / 4, kb = blk % 4;
    const __nv_bfloat162 *xr = x + ((long)n * in_f + blk * 32) / 2;  // 16 bf16x2 = 32 elems
    float2 v[16]; float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        v[i] = __bfloat1622float2(xr[i]);
        amax = fmaxf(amax, fmaxf(fabsf(v[i].x), fabsf(v[i].y)));
    }
    uint8_t sc = enc_ue4m3(amax * (1.f / 6.f));
    scaleB[((long)step * batch + n) * 4 + kb] = sc;
    float inv = 1.f / dec_ue4m3(sc);
    uint8_t *out = Bbytes + (long)n * (in_f / 2) + blk * 16;
#pragma unroll
    for (int i = 0; i < 16; i++)
        out[i] = q_fp4(v[i].x * inv) | (q_fp4(v[i].y * inv) << 4);
}
extern "C" void quantize_act_nvfp4(const void *x, void *Bbytes, void *scaleB, int batch, int in_f) {
    int total = batch * (in_f / 32), tpb = 256;
    quant_act<<<(total + tpb - 1) / tpb, tpb>>>((const __nv_bfloat162 *)x, (uint8_t *)Bbytes,
                                                (uint8_t *)scaleB, batch, in_f);
}

// ---- PyTorch-callable entry: raw device pointers (torch data_ptr) -> launch ----
extern "C" int sparse_fp4_mm(const void *A, const void *B, const void *scaleA,
                             const void *scaleB, const void *meta, void *C,
                             int M, int N, int Klog) {
    int KAb = Klog / 4, KBb = Klog / 2;
    alignas(64) CUtensorMap mapA, mapB;
    mk(&mapA, (uint8_t *)A, KAb, M, AW, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    mk(&mapB, (uint8_t *)B, KBb, N, BW_, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    cudaFuncSetAttribute(matmul_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / BN, M / (2 * BM)), block(256);
    matmul_sp<<<grid, block, SMEM>>>(mapA, mapB, (const uint8_t *)scaleA, (const uint8_t *)scaleB,
                                     (const uint32_t *)meta, (__nv_bfloat16 *)C, M, N, Klog);
    return (int)cudaDeviceSynchronize();
}
