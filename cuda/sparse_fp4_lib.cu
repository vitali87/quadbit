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
          __nv_bfloat16 *C, int M, int N, int Klog,
          const float *gA, const float *gB,     // per-row(M)/per-col(N) fp32 global (two-level NVFP4);
                                                 // nullptr -> single-level (mma-applied local scales only)
          int outT) {                            // outT: 0 -> C is [M,N]; 1 -> C is [N,M] (token-major, contiguous for vLLM)
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
    if (!outT) {
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int idx = mt * 8 + n;
                int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + b_col[n] + (lane & 3) * 2;
                float ga0 = gA ? gA[gr] : 1.f, ga1 = gA ? gA[gr + 8] : 1.f;   // two-level global rescale:
                float gb0 = gB ? gB[gc] : 1.f, gb1 = gB ? gB[gc + 1] : 1.f;   // mma applied locals, globals here
                *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0] * ga0 * gb0, acc[idx][1] * ga0 * gb1);
                *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2] * ga1 * gb0, acc[idx][3] * ga1 * gb1);
            }
    } else {
        // Transposed epilogue (zero-copy): stage the [2*BM out_f x BN token] tile in smem token-major,
        // then write C as [N token, M out_f] row-major so consecutive threads hit consecutive out_f ->
        // coalesced global stores AND a contiguous tensor for vLLM (no separate transpose+copy pass).
        // Reuses the (dead post-loop) input smem: tile = BN*2*BM bf16 = 64KB <= SMEM.
        __nv_bfloat16 *Cs = (__nv_bfloat16 *)smem;   // Cs[lc*(2*BM) + lr]: lc=token 0..BN-1, lr=out_f 0..2*BM-1
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int idx = mt * 8 + n;
                int lr = wg * BM + a_rowt[mt] + (lane >> 2);          // local out_f
                int lc = b_col[n] + (lane & 3) * 2;                   // local token
                int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + lc;
                float ga0 = gA ? gA[gr] : 1.f, ga1 = gA ? gA[gr + 8] : 1.f;
                float gb0 = gB ? gB[gc] : 1.f, gb1 = gB ? gB[gc + 1] : 1.f;
                Cs[lc * (2 * BM) + lr] = __float2bfloat16_rn(acc[idx][0] * ga0 * gb0);
                Cs[(lc + 1) * (2 * BM) + lr] = __float2bfloat16_rn(acc[idx][1] * ga0 * gb1);
                Cs[lc * (2 * BM) + lr + 8] = __float2bfloat16_rn(acc[idx][2] * ga1 * gb0);
                Cs[(lc + 1) * (2 * BM) + lr + 8] = __float2bfloat16_rn(acc[idx][3] * ga1 * gb1);
            }
        __syncthreads();
        int base_m = blockIdx.y * (2 * BM), base_n = blockIdx.x * BN;
#pragma unroll
        for (int col = 0; col < BN; col++)
            C[(size_t)(base_n + col) * M + base_m + tid] = Cs[col * (2 * BM) + tid];
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
__global__ void quant_act(const int4 *x4, uint32_t *Bwords, uint8_t *scaleB, int batch, int in_f) {
    int b32 = in_f / 32;                               // 32-elem blocks per row
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * b32) return;
    int n = t / b32, blk = t % b32, step = blk / 4, kb = blk % 4;
    long base = (long)n * (in_f / 8) + blk * 4;        // 4 int4 = 32 bf16 = one 32-block
    float2 v[16]; float amax = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {                      // 128-bit coalesced loads
        int4 p = x4[base + q];
        const __nv_bfloat162 *bp = (const __nv_bfloat162 *)&p;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[q * 4 + j] = __bfloat1622float2(bp[j]);
            amax = fmaxf(amax, fmaxf(fabsf(v[q * 4 + j].x), fabsf(v[q * 4 + j].y)));
        }
    }
    uint8_t sc = enc_ue4m3(amax * (1.f / 6.f));
    scaleB[((long)step * batch + n) * 4 + kb] = sc;
    float inv = 1.f / dec_ue4m3(sc);
    uint32_t w[4] = {0, 0, 0, 0};                       // 16 packed bytes = 4 u32 (128-bit store)
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t byte = q_fp4(v[i].x * inv) | (q_fp4(v[i].y * inv) << 4);
        w[i >> 2] |= byte << ((i & 3) * 8);
    }
    uint4 out = make_uint4(w[0], w[1], w[2], w[3]);
    *reinterpret_cast<uint4 *>(Bwords + (long)n * (in_f / 8) + blk * 4) = out;
}
extern "C" void quantize_act_nvfp4(const void *x, void *Bbytes, void *scaleB, int batch, int in_f) {
    int total = batch * (in_f / 32), tpb = 256;
    quant_act<<<(total + tpb - 1) / tpb, tpb>>>((const int4 *)x, (uint32_t *)Bbytes,
                                                (uint8_t *)scaleB, batch, in_f);
}

// TWO-LEVEL activation quantizer: per-token fp32 global gB[n]=rowamax/2688, per-32 ue4m3 LOCAL
// relative to gB (the sparse mma's scale_vec::4X gives 4 scales/128-K on B -> per-32). One CTA/row.
// scaleB[step][batch][4] LOCAL codes; the two-level kernel's epilogue applies gB[n].
__global__ void quant_act_2lvl_k(const __nv_bfloat16 *x, uint32_t *Bwords, uint8_t *scaleB, float *gB, int in_f) {
    int n = blockIdx.x;
    const __nv_bfloat16 *xr = x + (long)n * in_f;
    float la = 0.f;
    for (int i = threadIdx.x; i < in_f; i += blockDim.x) la = fmaxf(la, fabsf(__bfloat162float(xr[i])));
    for (int o = 16; o > 0; o >>= 1) la = fmaxf(la, __shfl_down_sync(0xffffffffu, la, o));
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = la;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0.f; int nw = blockDim.x >> 5; for (int i = 0; i < nw; i++) t = fmaxf(t, red[i]); red[0] = t; }
    __syncthreads();
    float rowamax = red[0];
    float g = rowamax > 0.f ? rowamax / 2688.f : 1.f;      // 2688 = e4m3max(448) * e2m1max(6)
    if (threadIdx.x == 0) gB[n] = g;
    int b32 = in_f / 32;
    for (int blk = threadIdx.x; blk < b32; blk += blockDim.x) {
        int step = blk / 4, kb = blk % 4;
        float v[32], amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) { v[i] = __bfloat162float(xr[blk * 32 + i]); amax = fmaxf(amax, fabsf(v[i])); }
        uint8_t code = enc_ue4m3((amax / 6.f) / g);        // local relative to global -> e4m3 sweet spot
        scaleB[((long)step * gridDim.x + n) * 4 + kb] = code;
        float inv = 1.f / (dec_ue4m3(code) * g);
        uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t byte = q_fp4(v[2 * i] * inv) | (q_fp4(v[2 * i + 1] * inv) << 4);
            w[i >> 2] |= byte << ((i & 3) * 8);
        }
        *reinterpret_cast<uint4 *>(Bwords + (long)n * (in_f / 8) + blk * 4) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}
extern "C" void quantize_act_nvfp4_2lvl(const void *x, void *Bbytes, void *scaleB, void *gB, int batch, int in_f) {
    quant_act_2lvl_k<<<batch, 256>>>((const __nv_bfloat16 *)x, (uint32_t *)Bbytes, (uint8_t *)scaleB,
                                     (float *)gB, in_f);
}

// ---- Fused SwiGLU epilogue: silu(g)*u + NVFP4 quantize in ONE pass. g,u are the gate/up GEMM
// outputs in [hidden, batch] (the kernel's C[out,batch] layout); output Hbytes[batch,hidden/2] +
// scaleH[hidden/128, batch, 4] is the down-proj's activation input. Replaces eager silu+mul+2
// casts + a separate transpose + a separate quant pass (~5 memory round-trips over [batch,hidden])
// with a single fused kernel. One thread owns one (batch b, 32-block of hidden); consecutive
// threads take consecutive b so the strided g/u reads coalesce across the warp.
__global__ void swiglu_quant(const __nv_bfloat16 *g, const __nv_bfloat16 *u, uint32_t *Hwords,
                             uint8_t *scaleH, int batch, int hidden) {
    int hb = hidden / 32;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * hb) return;
    int hblock = (int)(t / batch), b = (int)(t % batch);   // b fastest -> coalesced g/u reads
    int step = hblock / 4, kb = hblock % 4;
    long base = (long)(hblock * 32) * batch + b;
    float val[32]; float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        float gv = __bfloat162float(g[base + (long)i * batch]);
        float uv = __bfloat162float(u[base + (long)i * batch]);
        float h = (gv / (1.f + __expf(-gv))) * uv;         // silu(g)*u
        val[i] = h; amax = fmaxf(amax, fabsf(h));
    }
    uint8_t sc = enc_ue4m3(amax * (1.f / 6.f));
    scaleH[((long)step * batch + b) * 4 + kb] = sc;
    float inv = 1.f / dec_ue4m3(sc);
    uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t byte = q_fp4(val[2 * i] * inv) | (q_fp4(val[2 * i + 1] * inv) << 4);
        w[i >> 2] |= byte << ((i & 3) * 8);
    }
    *reinterpret_cast<uint4 *>(Hwords + (long)b * (hidden / 8) + hblock * 4) = make_uint4(w[0], w[1], w[2], w[3]);
}
extern "C" void fused_swiglu_quant(const void *g, const void *u, void *Hbytes, void *scaleH,
                                   int batch, int hidden) {
    int total = batch * (hidden / 32), tpb = 256;
    swiglu_quant<<<(total + tpb - 1) / tpb, tpb>>>((const __nv_bfloat16 *)g, (const __nv_bfloat16 *)u,
                                                   (uint32_t *)Hbytes, (uint8_t *)scaleH, batch, hidden);
}

// TWO-LEVEL fused SwiGLU (emits per-token global gH so the down GEMM runs two-level, gB=gH, not
// single-level gB=1 -- closes the ~11 vs 8.95 fused-path accuracy gap). Two COALESCED passes over
// Cgu[hidden,batch] (consecutive threads = consecutive tokens b, so each e-step's loads coalesce -- the
// CTA-per-token layout reads strided-by-batch and is ~25% slower). Pass 1 = per-token amax: one thread
// owns one token, loops the hidden dim, reduces its OWN token's amax in a register (no atomics, no
// cross-thread reduce) -> gH. Pass 2 = the fast single-level quantizer, but encoded RELATIVE to gH.
// Pass 1: per-token amax via the FAST single-level layout (one thread per 32-block, massively parallel
// -> good at prefill AND decode, unlike a one-thread-per-token serial loop). Each thread reduces its
// block's amax and atomicMaxes it into gH[b] (float>=0, so __float_as_int is monotonic; gH pre-zeroed).
__global__ void swiglu_amax(const __nv_bfloat16 *g, const __nv_bfloat16 *u, float *gH, int batch, int hidden) {
    int hb = hidden / 32;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * hb) return;
    int hblock = (int)(t / batch), b = (int)(t % batch);   // b fastest -> coalesced g/u reads
    long base = (long)(hblock * 32) * batch + b;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        float gv = __bfloat162float(g[base + (long)i * batch]);
        float uv = __bfloat162float(u[base + (long)i * batch]);
        amax = fmaxf(amax, fabsf((gv / (1.f + __expf(-gv))) * uv));
    }
    atomicMax((int *)&gH[b], __float_as_int(amax));
}
__global__ void swiglu_finalize(float *gH, int batch) {    // raw per-token amax -> global (amax/2688)
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    gH[b] = gH[b] > 0.f ? gH[b] / 2688.f : 1.f;            // 2688 = e4m3max(448) * e2m1max(6)
}
__global__ void swiglu_quant_g(const __nv_bfloat16 *g, const __nv_bfloat16 *u, uint32_t *Hwords,
                               uint8_t *scaleH, const float *gH, int batch, int hidden) {
    int hb = hidden / 32;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)batch * hb) return;
    int hblock = (int)(t / batch), b = (int)(t % batch);   // b fastest -> coalesced g/u reads
    int step = hblock / 4, kb = hblock % 4;
    long base = (long)(hblock * 32) * batch + b;
    float gg = gH[b];
    float val[32], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        float gv = __bfloat162float(g[base + (long)i * batch]);
        float uv = __bfloat162float(u[base + (long)i * batch]);
        float h = (gv / (1.f + __expf(-gv))) * uv;
        val[i] = h; amax = fmaxf(amax, fabsf(h));
    }
    uint8_t code = enc_ue4m3((amax / 6.f) / gg);           // local relative to per-token global
    scaleH[((long)step * batch + b) * 4 + kb] = code;
    float inv = 1.f / (dec_ue4m3(code) * gg);
    uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t byte = q_fp4(val[2 * i] * inv) | (q_fp4(val[2 * i + 1] * inv) << 4);
        w[i >> 2] |= byte << ((i & 3) * 8);
    }
    *reinterpret_cast<uint4 *>(Hwords + (long)b * (hidden / 8) + hblock * 4) = make_uint4(w[0], w[1], w[2], w[3]);
}
// gH MUST be pre-zeroed by the caller (swiglu_amax atomicMaxes into it).
extern "C" void fused_swiglu_quant_2lvl(const void *g, const void *u, void *Hbytes, void *scaleH,
                                        void *gH, int batch, int hidden) {
    int tpb = 256, total = batch * (hidden / 32);
    swiglu_amax<<<(total + tpb - 1) / tpb, tpb>>>((const __nv_bfloat16 *)g, (const __nv_bfloat16 *)u,
                                                  (float *)gH, batch, hidden);
    swiglu_finalize<<<(batch + tpb - 1) / tpb, tpb>>>((float *)gH, batch);
    swiglu_quant_g<<<(total + tpb - 1) / tpb, tpb>>>((const __nv_bfloat16 *)g, (const __nv_bfloat16 *)u,
                                                     (uint32_t *)Hbytes, (uint8_t *)scaleH, (const float *)gH, batch, hidden);
}

// ---- Fused RMSNorm + NVFP4 quantize: the transformer block entry. Reads the residual-stream row
// x[b,:], computes the RMS over `hidden`, normalizes x*rms*weight, and quantizes per-32-block to
// FP4 + ue4m3 scales -- output feeds straight into the QKV / gate-up GEMM. One CTA per row loads
// the row into smem, block-reduces the sum-of-squares, then each thread quantizes its 32-blocks
// from smem. Replaces eager rmsnorm (read+reduce+write) + a separate quant pass (read+write) with a
// single read of x; also more accurate (no bf16 round-trip of the normalized value before quant).
__global__ void rmsnorm_quant_k(const __nv_bfloat16 *x, const __nv_bfloat16 *wt, uint32_t *Bwords,
                                uint8_t *scaleB, int batch, int hidden, float eps) {
    int b = blockIdx.x;
    extern __shared__ float sm[];                     // the row, in float
    const __nv_bfloat16 *xr = x + (long)b * hidden;
    float ls = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(xr[i]); sm[i] = v; ls += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ls += __shfl_down_sync(0xffffffffu, ls, o);
    __shared__ float red[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = ls;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; i++) t += red[i];
        red[0] = rsqrtf(t / hidden + eps);
    }
    __syncthreads();
    float rms = red[0];
    int nb = hidden / 32;
    for (int blk = threadIdx.x; blk < nb; blk += blockDim.x) {
        int step = blk / 4, kb = blk % 4;
        float val[32]; float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            float v = sm[blk * 32 + i] * rms * __bfloat162float(wt[blk * 32 + i]);
            val[i] = v; amax = fmaxf(amax, fabsf(v));
        }
        uint8_t sc = enc_ue4m3(amax * (1.f / 6.f));
        scaleB[((long)step * batch + b) * 4 + kb] = sc;
        float inv = 1.f / dec_ue4m3(sc);
        uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t byte = q_fp4(val[2 * i] * inv) | (q_fp4(val[2 * i + 1] * inv) << 4);
            w[i >> 2] |= byte << ((i & 3) * 8);
        }
        *reinterpret_cast<uint4 *>(Bwords + (long)b * (hidden / 8) + blk * 4) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}
extern "C" void rmsnorm_quant(const void *x, const void *wt, void *Bbytes, void *scaleB,
                              int batch, int hidden, float eps) {
    int smem = hidden * (int)sizeof(float);
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(rmsnorm_quant_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    rmsnorm_quant_k<<<batch, 256, smem>>>((const __nv_bfloat16 *)x, (const __nv_bfloat16 *)wt,
                                          (uint32_t *)Bbytes, (uint8_t *)scaleB, batch, hidden, eps);
}

// ---- Fused residual-add + RMSNorm + NVFP4 quant: the full block transition. h = inp + residual
// (written back as the updated residual stream for the next add), then rmsnorm(h)*weight quantized
// to FP4 for the next GEMM. Fuses the eager residual add (read2+write) into the norm+quant kernel:
// one read of inp+residual replaces add(read2,write1) + rmsnorm(read,reduce,write) + quant(read,write).
__global__ void add_rmsnorm_quant_k(const __nv_bfloat16 *inp, const __nv_bfloat16 *res,
                                    const __nv_bfloat16 *wt, __nv_bfloat16 *hout, uint32_t *Bwords,
                                    uint8_t *scaleB, int batch, int hidden, float eps) {
    int b = blockIdx.x;
    extern __shared__ float sm[];
    const __nv_bfloat16 *ir = inp + (long)b * hidden;
    const __nv_bfloat16 *rr = res + (long)b * hidden;
    __nv_bfloat16 *ho = hout + (long)b * hidden;
    float ls = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(ir[i]) + __bfloat162float(rr[i]);
        sm[i] = v; ho[i] = __float2bfloat16_rn(v); ls += v * v;   // write updated residual stream
    }
    for (int o = 16; o > 0; o >>= 1) ls += __shfl_down_sync(0xffffffffu, ls, o);
    __shared__ float red[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = ls;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; i++) t += red[i];
        red[0] = rsqrtf(t / hidden + eps);
    }
    __syncthreads();
    float rms = red[0];
    int nb = hidden / 32;
    for (int blk = threadIdx.x; blk < nb; blk += blockDim.x) {
        int step = blk / 4, kb = blk % 4;
        float val[32]; float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            float v = sm[blk * 32 + i] * rms * __bfloat162float(wt[blk * 32 + i]);
            val[i] = v; amax = fmaxf(amax, fabsf(v));
        }
        uint8_t sc = enc_ue4m3(amax * (1.f / 6.f));
        scaleB[((long)step * batch + b) * 4 + kb] = sc;
        float inv = 1.f / dec_ue4m3(sc);
        uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t byte = q_fp4(val[2 * i] * inv) | (q_fp4(val[2 * i + 1] * inv) << 4);
            w[i >> 2] |= byte << ((i & 3) * 8);
        }
        *reinterpret_cast<uint4 *>(Bwords + (long)b * (hidden / 8) + blk * 4) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}
extern "C" void add_rmsnorm_quant(const void *inp, const void *res, const void *wt, void *hout,
                                  void *Bbytes, void *scaleB, int batch, int hidden, float eps) {
    int smem = hidden * (int)sizeof(float);
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(add_rmsnorm_quant_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    add_rmsnorm_quant_k<<<batch, 256, smem>>>((const __nv_bfloat16 *)inp, (const __nv_bfloat16 *)res,
                                              (const __nv_bfloat16 *)wt, (__nv_bfloat16 *)hout,
                                              (uint32_t *)Bbytes, (uint8_t *)scaleB, batch, hidden, eps);
}

// ---- PyTorch-callable entry: raw device pointers (torch data_ptr) -> launch ----
// Core launcher (NO sync): build TMA maps + launch matmul_sp on the current (per-thread) stream. The
// per-thread-stream .so binds <<<>>> to the caller's (vLLM's) stream, so kernels are ordered without an
// explicit sync -- used by fused_mlp_2lvl to batch the whole MLP with zero device syncs.
// cudaFuncSetAttribute is a host-side call that is ILLEGAL during CUDA-graph stream capture. Set the
// matmul_sp max-dynamic-smem ONCE (idempotent, SMEM is a compile-time constant) so run_sp_mm contains
// only stream operations (TMA-descriptor build on host + kernel launch) and is capture-safe thereafter.
static bool g_sp_attr_set = false;
extern "C" void qb_init_func_attrs() {
    cudaFuncSetAttribute(matmul_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    g_sp_attr_set = true;
}
static inline void run_sp_mm(const void *A, const void *B, const void *scaleA, const void *scaleB,
                             const void *meta, void *C, int M, int N, int Klog,
                             const void *gA, const void *gB, int outT, cudaStream_t stream) {
    int KAb = Klog / 4, KBb = Klog / 2;
    alignas(64) CUtensorMap mapA, mapB;
    mk(&mapA, (uint8_t *)A, KAb, M, AW, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    mk(&mapB, (uint8_t *)B, KBb, N, BW_, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    if (!g_sp_attr_set) qb_init_func_attrs();   // one-time; skipped during capture (set at warmup)
    dim3 grid(N / BN, M / (2 * BM)), block(256);
    matmul_sp<<<grid, block, SMEM, stream>>>(mapA, mapB, (const uint8_t *)scaleA, (const uint8_t *)scaleB,
                                             (const uint32_t *)meta, (__nv_bfloat16 *)C, M, N, Klog,
                                             (const float *)gA, (const float *)gB, outT);
}
extern "C" int sparse_fp4_mm(const void *A, const void *B, const void *scaleA,
                             const void *scaleB, const void *meta, void *C, int M, int N, int Klog) {
    run_sp_mm(A, B, scaleA, scaleB, meta, C, M, N, Klog, nullptr, nullptr, 0, 0);
    return (int)cudaDeviceSynchronize();
}
// TWO-LEVEL entry: same kernel, plus per-row(M=weight-row) gA and per-col(N=token) gB fp32 globals.
// scaleA/scaleB now hold ue4m3 codes LOCAL to the globals; the epilogue applies gA[m]*gB[n].
extern "C" int sparse_fp4_mm_2lvl(const void *A, const void *B, const void *scaleA, const void *scaleB,
                                  const void *meta, void *C, int M, int N, int Klog, const void *gA, const void *gB) {
    run_sp_mm(A, B, scaleA, scaleB, meta, C, M, N, Klog, gA, gB, 0, 0);
    return (int)cudaDeviceSynchronize();
}
// ZERO-COPY entry: identical math to _2lvl, but writes C in [N token, M out_f] row-major (outT=1)
// so the caller returns C[:t] directly to vLLM with NO transpose+copy pass. See matmul_sp epilogue.
extern "C" int sparse_fp4_mm_2lvl_t(const void *A, const void *B, const void *scaleA, const void *scaleB,
                                    const void *meta, void *C, int M, int N, int Klog, const void *gA, const void *gB) {
    run_sp_mm(A, B, scaleA, scaleB, meta, C, M, N, Klog, gA, gB, 1, 0);
    return (int)cudaDeviceSynchronize();
}

// ---- FUSED MLP: the WHOLE two-level sparse SwiGLU MLP in ONE ctypes entry, ZERO device syncs. Collapses
// the Python-driven 6-crossings + 2 cudaDeviceSynchronize-per-layer (64 syncs/forward) into a single call
// whose 6 kernels stream back-to-back on vLLM's per-thread stream. gate_up mma (outT=0, [gu_out,tp]) ->
// two-level swiglu (amax/finalize/quant_g, emits gH) -> down mma (outT=1, [tp,dn_out] contiguous for vLLM).
// All scratch is caller-preallocated (graph-capture-safe: no cudaMalloc/free, no sync in the region). gH
// MUST be pre-zeroed by the caller (swiglu_amax atomicMaxes into it).
// `stream` (raw cudaStream_t as void*; 0 = default) makes this CUDA-GRAPH-CAPTURABLE: every kernel and
// the gH zeroing run on the CALLER's current stream (torch/vLLM's capture stream), and there are no host
// syncs, no allocations, and no host reads of device values in the region. gH is zeroed here (cudaMemset
// Async, capturable) so the caller need not pre-zero it -- swiglu_amax atomicMaxes into it.
extern "C" int fused_mlp_2lvl(const void *x, const void *gu_Ac, const void *gu_scaleA, const void *gu_meta,
                              const void *gu_gA, const void *dn_Ac, const void *dn_scaleA, const void *dn_meta,
                              const void *dn_gA, void *Bb, void *sBg, void *gBg, void *Cgu, void *Hb,
                              void *sH, void *gH, void *Cout, int tp, int H, int Iw, int gu_out, int dn_out,
                              void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int tpb = 256;
    cudaMemsetAsync(gH, 0, (size_t)tp * sizeof(float), s);   // capturable zero; swiglu_amax atomicMaxes into gH
    quant_act_2lvl_k<<<tp, tpb, 0, s>>>((const __nv_bfloat16 *)x, (uint32_t *)Bb, (uint8_t *)sBg, (float *)gBg, H);
    run_sp_mm(gu_Ac, Bb, gu_scaleA, sBg, gu_meta, Cgu, gu_out, tp, H, gu_gA, gBg, 0, s);
    const __nv_bfloat16 *g = (const __nv_bfloat16 *)Cgu, *uu = g + (long)Iw * tp;   // Cgu is [gu_out=2*Iw, tp]
    int total = tp * (Iw / 32);
    swiglu_amax<<<(total + tpb - 1) / tpb, tpb, 0, s>>>(g, uu, (float *)gH, tp, Iw);
    swiglu_finalize<<<(tp + tpb - 1) / tpb, tpb, 0, s>>>((float *)gH, tp);
    swiglu_quant_g<<<(total + tpb - 1) / tpb, tpb, 0, s>>>(g, uu, (uint32_t *)Hb, (uint8_t *)sH, (const float *)gH, tp, Iw);
    run_sp_mm(dn_Ac, Hb, dn_scaleA, sH, dn_meta, Cout, dn_out, tp, Iw, dn_gA, gH, 1, s);
    return 0;
}
