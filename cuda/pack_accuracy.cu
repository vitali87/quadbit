// Raw-PTX track: FULLY DEPLOYABLE 2:4-sparse FP4 matmul -- arbitrary per-group 2:4
// metadata + real per-block ue4m3 scales, both staged coalesced through the async
// pipeline. Combines every derived+verified layout:
//   scaleA[row r][kb]->lane (r&7)*4+(r>>3) byte kb ; scaleB[col c][kb]->lane c*4 byte kb
//   metadata: lane L of an m-tile -> mma-row (L&1)*8+(L>>2), half H=(L>>1)&1; e = 8 nibbles
//     for groups [H*8..H*8+8); nibble = idx0|(idx1<<2) (kept pair-indices of each 4).
// Tensors are STEP-MAJOR so each step's CTA slice is contiguous and bulk-copied to smem:
//   scaleA [ksteps][M][4], scaleB [ksteps][N][4], meta [ksteps][M][2] u32. STAGES=5.

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#define BM 128
#define BN 128
#define BKL 128
#define AROWB 32
#define BROWB 64
#define STAGES 5
#define ASZ (BM * AROWB)
#define BSZ (BN * BROWB)
#define SCA 1024          // scaleA slice (256 CTA rows x 4)
#define SCB 512           // scaleB slice (128 CTA cols x 4)
#define MET 2048          // metadata slice (256 CTA rows x 2 u32)
#define SMEM (2 * STAGES * ASZ + STAGES * BSZ + STAGES * SCA + STAGES * SCB + STAGES * MET + 2 * STAGES * 8 + 128)

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
    uint8_t *scB_sm = scA_sm + STAGES * SCA;
    uint8_t *met_sm = scB_sm + STAGES * SCB;
    uint64_t *full = (uint64_t *)(met_sm + STAGES * MET);
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

    // scale smem-read indices (ue4m3 4X layout)
    int ra_local = (lane & 3) * 8 + (lane >> 2), cb_local = lane >> 2;
    bool a_valid = ra_local < 16, b_valid = (lane & 3) == 0;
    int a_sidx[4], b_sidx[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_sidx[mt] = wg * 128 + a_rowt[mt] + ra_local;
#pragma unroll
    for (int n = 0; n < 8; n++) b_sidx[n] = b_col[n] + cb_local;
    // metadata smem-read indices: lane -> mma-row (lane&1)*8+(lane>>2), half (lane>>1)&1
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

    auto issue = [&](int s, int step) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                     "r"((uint32_t)(2 * ASZ + BSZ + SCA + SCB + MET)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])),
                     "l"(&mapA), "r"(step * AROWB), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])),
                     "l"(&mapB), "r"(step * BROWB), "r"(block_col), "r"(bar));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&scA_sm[s * SCA])),
                     "l"(scaleA + (size_t)(step * M + a_load_row) * 4), "r"((uint32_t)SCA), "r"(bar));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&scB_sm[s * SCB])),
                     "l"(scaleB + (size_t)(step * N + block_col) * 4), "r"((uint32_t)SCB), "r"(bar));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&met_sm[s * MET])),
                     "l"((const uint8_t *)meta + (size_t)(step * M + a_load_row) * 8), "r"((uint32_t)MET), "r"(bar));
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

        const uint32_t *scA = (const uint32_t *)(scA_sm + s * SCA);
        const uint32_t *scB = (const uint32_t *)(scB_sm + s * SCB);
        const uint32_t *mtA = (const uint32_t *)(met_sm + s * MET);
        uint32_t sav[4], sbv[8], ev[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) { sav[mt] = a_valid ? scA[a_sidx[mt]] : 0x38383838u; ev[mt] = mtA[m_sidx[mt]]; }
#pragma unroll
        for (int n = 0; n < 8; n++) sbv[n] = b_valid ? scB[b_sidx[n]] : 0x38383838u;

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
                      "r"(ev[mt]), "r"(sav[mt]), "h"(z), "h"(z), "r"(sbv[n]), "h"(z), "h"(z));
                acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
            }
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

static float dfp4(uint8_t n){int s=(n>>3)&1,e=(n>>1)&3,m=n&1;float v=(e==0)?(m?0.5f:0.f):((float)(1<<(e-1))*(1.f+0.5f*m));return s?-v:v;}
static float due4m3(uint8_t n){int s=(n>>7)&1,e=(n>>3)&0xf,m=n&7;float v=(e==0)?((float)m*0.001953125f):(float)(1+m/8.0)*exp2f((float)(e-7));return s?-v:v;}

#include <math.h>
// nearest e2m1 (fp4) code for |x|; levels {0,.5,1,1.5,2,3,4,6} -> codes 0..7
static uint8_t q_fp4(float x) {
    static const float lv[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    float a = fabsf(x); int best = 0; float bd = 1e30f;
    for (int c = 0; c < 8; c++) { float d = fabsf(a - lv[c]); if (d < bd) { bd = d; best = c; } }
    return (uint8_t)(best | (x < 0 ? 8 : 0));
}
// nearest ue4m3 byte for scale s>0
static uint8_t q_ue4m3(float s) {
    int best = 0x38; float bd = 1e30f;
    for (int e = 0; e < 16; e++) for (int m = 0; m < 8; m++) {
        uint8_t b = (e << 3) | m; float v = due4m3(b); float d = fabsf(s - v);
        if (d < bd) { bd = d; best = b; }
    }
    return (uint8_t)best;
}

// REAL-quantization accuracy study: true fp32 dense weights -> 2:4-prune + NVFP4 quantize
// (with rounding) -> kernel; compares kernel vs CPU-emulated sparse-FP4 (validates the
// kernel on genuinely-rounded data) AND emulated vs true dense fp32 (the accuracy cost).
int main() {
    auto go = [](int sz) {
        int M = sz, N = sz, Klog = sz, KAb = Klog / 4, KBb = Klog / 2, ksteps = Klog / BKL;
        float *Ad = (float *)malloc((size_t)M * Klog * 4), *Bd = (float *)malloc((size_t)N * Klog * 4);
        uint32_t st = 0x5u; auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
        auto frand = [&]() { return ((float)(rnd() % 20001) / 10000.f) - 1.f; };  // [-1,1]
        for (size_t i = 0; i < (size_t)M * Klog; i++) Ad[i] = frand();
        for (size_t i = 0; i < (size_t)N * Klog; i++) Bd[i] = frand();

        uint8_t *hAc = (uint8_t *)calloc((size_t)M * KAb, 1), *hB = (uint8_t *)malloc((size_t)N * KBb);
        uint8_t *hScA = (uint8_t *)malloc((size_t)ksteps * M * 4), *hScB = (uint8_t *)malloc((size_t)ksteps * N * 4);
        uint32_t *hMeta = (uint32_t *)calloc((size_t)ksteps * M * 2, 4);
        uint8_t *idxA = (uint8_t *)malloc((size_t)M * ksteps * 16 * 2);
        float *Aq = (float *)calloc((size_t)M * Klog, 4), *Bq = (float *)malloc((size_t)N * Klog * 4);

        // ---- B: quantize per 32-element block (scaleB layout: full-K/32) ----
        for (int j = 0; j < N; j++)
            for (int step = 0; step < ksteps; step++)
                for (int kb = 0; kb < 4; kb++) {
                    int k0 = step * 128 + kb * 32; float mx = 0;
                    for (int t = 0; t < 32; t++) { float a = fabsf(Bd[(size_t)j * Klog + k0 + t]); if (a > mx) mx = a; }
                    uint8_t sc = q_ue4m3(mx > 0 ? mx / 6.f : 1.f); float sv = due4m3(sc);
                    hScB[((size_t)step * N + j) * 4 + kb] = sc;
                    for (int t = 0; t < 32; t++) {
                        int k = k0 + t; uint8_t c = q_fp4(Bd[(size_t)j * Klog + k] / sv);
                        Bq[(size_t)j * Klog + k] = dfp4(c) * sv;
                        uint8_t *bp = &hB[(size_t)j * KBb + k / 2];
                        if (k & 1) *bp = (*bp & 0x0f) | (c << 4); else *bp = (*bp & 0xf0) | c;
                    }
                }

        // ---- A: 2:4 prune by magnitude per group of 4 pairs, then quantize kept per 16-nonzero block ----
        for (int i = 0; i < M; i++)
            for (int step = 0; step < ksteps; step++) {
                // first decide kept indices per group (by pair magnitude in fp32)
                for (int g = 0; g < 16; g++) {
                    float pm[4];
                    for (int p = 0; p < 4; p++) {
                        int pair = step * 64 + g * 4 + p;
                        pm[p] = fabsf(Ad[(size_t)i * Klog + 2 * pair]) + fabsf(Ad[(size_t)i * Klog + 2 * pair + 1]);
                    }
                    int i0 = 0; for (int p = 1; p < 4; p++) if (pm[p] > pm[i0]) i0 = p;
                    int i1 = -1; for (int p = 0; p < 4; p++) if (p != i0 && (i1 < 0 || pm[p] > pm[i1])) i1 = p;
                    if (i0 > i1) { int t = i0; i0 = i1; i1 = t; }
                    idxA[(((size_t)i * ksteps + step) * 16 + g) * 2 + 0] = i0;
                    idxA[(((size_t)i * ksteps + step) * 16 + g) * 2 + 1] = i1;
                    hMeta[((size_t)step * M + i) * 2 + g / 8] |= (uint32_t)(i0 | (i1 << 2)) << ((g % 8) * 4);
                }
                // scale per compressed 16-nonzero block (= 4 groups); then quantize kept values
                for (int kb = 0; kb < 4; kb++) {
                    float mx = 0;
                    for (int gg = 0; gg < 4; gg++) { int g = kb * 4 + gg;
                        for (int s = 0; s < 2; s++) { int pidx = idxA[(((size_t)i*ksteps+step)*16+g)*2+s]; int pair = step*64+g*4+pidx;
                            float a0 = fabsf(Ad[(size_t)i*Klog+2*pair]), a1 = fabsf(Ad[(size_t)i*Klog+2*pair+1]);
                            if (a0 > mx) mx = a0; if (a1 > mx) mx = a1; } }
                    uint8_t sc = q_ue4m3(mx > 0 ? mx / 6.f : 1.f); float sv = due4m3(sc);
                    hScA[((size_t)step * M + i) * 4 + kb] = sc;
                    for (int gg = 0; gg < 4; gg++) { int g = kb * 4 + gg;
                        for (int s = 0; s < 2; s++) { int pidx = idxA[(((size_t)i*ksteps+step)*16+g)*2+s]; int pair = step*64+g*4+pidx;
                            uint8_t clo = q_fp4(Ad[(size_t)i*Klog+2*pair] / sv), chi = q_fp4(Ad[(size_t)i*Klog+2*pair+1] / sv);
                            int cs = step*32 + g*2 + s; hAc[(size_t)i*KAb+cs] = clo | (chi << 4);
                            Aq[(size_t)i*Klog+2*pair] = dfp4(clo)*sv; Aq[(size_t)i*Klog+2*pair+1] = dfp4(chi)*sv;
                        } }
                }
            }

        uint8_t *dA,*dB,*dScA,*dScB; uint32_t *dMeta; __nv_bfloat16 *dC;
        cudaMalloc(&dA,(size_t)M*KAb);cudaMalloc(&dB,(size_t)N*KBb);
        cudaMalloc(&dScA,(size_t)ksteps*M*4);cudaMalloc(&dScB,(size_t)ksteps*N*4);
        cudaMalloc(&dMeta,(size_t)ksteps*M*2*4);cudaMalloc(&dC,(size_t)M*N*sizeof(__nv_bfloat16));
        cudaMemcpy(dA,hAc,(size_t)M*KAb,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,(size_t)N*KBb,cudaMemcpyHostToDevice);
        cudaMemcpy(dScA,hScA,(size_t)ksteps*M*4,cudaMemcpyHostToDevice);cudaMemcpy(dScB,hScB,(size_t)ksteps*N*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dMeta,hMeta,(size_t)ksteps*M*2*4,cudaMemcpyHostToDevice);
        alignas(64) CUtensorMap mapA,mapB; mk("A",&mapA,dA,KAb,M,AROWB,BM); mk("B",&mapB,dB,KBb,N,BROWB,BN);
        cudaFuncSetAttribute(matmul_sp,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
        dim3 grid(N/BN,M/(2*BM)),block(256);
        matmul_sp<<<grid,block,SMEM>>>(mapA,mapB,dScA,dScB,dMeta,dC,M,N,Klog);
        if(cudaDeviceSynchronize()!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(cudaGetLastError()));return;}
        __nv_bfloat16 *hC=(__nv_bfloat16*)malloc((size_t)M*N*sizeof(__nv_bfloat16));
        cudaMemcpy(hC,dC,(size_t)M*N*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost);

        // errors: kernel vs emulated sparse-FP4 (Aq*Bq), and emulated vs true dense (Ad*Bd)
        double ek=0,ee=0,nt=0; float maxk=0;
        for (int i=0;i<M;i++)for(int j=0;j<N;j++){
            float emul=0, tru=0;
            for(int k=0;k<Klog;k++){ emul += Aq[(size_t)i*Klog+k]*Bq[(size_t)j*Klog+k]; tru += Ad[(size_t)i*Klog+k]*Bd[(size_t)j*Klog+k]; }
            float got=__bfloat162float(hC[(size_t)i*N+j]);
            ek += (double)(got-emul)*(got-emul); ee += (double)(emul-tru)*(emul-tru); nt += (double)tru*tru;
            float rk=fabsf(got-emul)/(fabsf(emul)+1.f); if(rk>maxk)maxk=rk;
        }
        printf("pack_accuracy %dx%dx%d: kernel-vs-emul rel %.5f (max %.5f) [validates kernel] | emul-vs-true rel %.4f [2:4+FP4 cost]\n",
               sz,sz,sz, sqrt(ek/(nt+1e-9)), maxk, sqrt(ee/(nt+1e-9)));
        free(Ad);free(Bd);free(hAc);free(hB);free(hScA);free(hScB);free(hMeta);free(idxA);free(Aq);free(Bq);free(hC);
        cudaFree(dA);cudaFree(dB);cudaFree(dScA);cudaFree(dScB);cudaFree(dMeta);cudaFree(dC);
    };
    go(512); go(1024);
    return 0;
}
